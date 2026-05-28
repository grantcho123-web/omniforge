"""Simulated trading as a one-shot scoring task.

The v0.1 ``SingleAssetTradingEnv`` is a sequential decision-making
environment — you step it many times. Frontier-lab evals (the v0.2
product) ask a different question: *given a market snapshot, what
position would your model take?* This module bridges the two by
generating a ``Task`` from a market window. The LLM produces a single
position decision; the grader evaluates that position against the
realized next-bar return.

Why this matters: it preserves all the v0.1 trading work as a task
type that fits the new architecture, without requiring sequential
training. Customers who want the full RL training loop can still use
the underlying env directly via ``ebit_gym.envs``.

Two pieces:

* :class:`SimulatedTradingTaskBuilder` — turns an OHLCV DataFrame into a
  sequence of ``Task`` objects, one per market window.
* :func:`score_position` — the scoring rule, exposed as a regular Python
  function so an ``llm_judge`` or custom grader can call it.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ebit_gym.core.task import (
    GraderSpec,
    Task,
    TaskMaterial,
    TaskMetadata,
)
from ebit_gym.data.sources import OHLCV_COLUMNS


@dataclass
class SimulatedTradingTaskBuilder:
    """Turns an OHLCV DataFrame into trading-decision tasks.

    Each emitted task shows the model a fixed-length OHLCV window
    (rendered as a compact table material) and asks for a single
    position in [-1, 1] where -1 is fully short, 0 is flat, +1 is
    fully long. The reference position is the **sign of the next bar's
    return** — i.e. the trade an oracle with perfect single-bar
    foresight would take. The grader uses a numeric tolerance match so
    "close enough" sized positions get partial credit.

    Args:
        data: OHLCV DataFrame, time-indexed.
        window_size: How many bars of history shown per task.
        stride: How many bars to advance between tasks. Default 1 = every bar.
        max_tasks: Cap on number of tasks emitted, useful for large frames.
        symbol: Asset label included in the task prompt + metadata.
        position_tolerance: How close a position must be to the oracle's
            sign to fully match. With sign-only oracle, 0.5 means
            "right side of the market gets credit; magnitude is bonus."
    """

    data: pd.DataFrame
    window_size: int = 32
    stride: int = 1
    max_tasks: int | None = None
    symbol: str = "SPY"
    position_tolerance: float = 0.5

    def __post_init__(self) -> None:
        missing = set(OHLCV_COLUMNS) - set(self.data.columns)
        if missing:
            raise ValueError(f"data missing required columns: {sorted(missing)}")
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if self.stride < 1:
            raise ValueError("stride must be at least 1")
        if len(self.data) <= self.window_size + 1:
            raise ValueError("data must be longer than window_size + 1")
        if not (0.0 < self.position_tolerance <= 2.0):
            raise ValueError("position_tolerance must be in (0, 2]")

    def __iter__(self) -> Iterator[Task]:
        close = self.data["close"].to_numpy(dtype=float)
        n = len(self.data)

        candidate_ends = range(self.window_size, n - 1, self.stride)
        for emitted, end_idx in enumerate(candidate_ends):
            if self.max_tasks is not None and emitted >= self.max_tasks:
                break

            window = self.data.iloc[end_idx - self.window_size : end_idx]
            price_t = float(close[end_idx - 1])
            price_t1 = float(close[end_idx])
            next_return = (price_t1 - price_t) / price_t

            yield self._task_for_window(
                window=window,
                end_idx=end_idx,
                oracle_return=next_return,
            )

    def tasks(self) -> list[Task]:
        return list(self)

    # -------------------------------------------------------------------- helpers

    def _task_for_window(
        self,
        window: pd.DataFrame,
        end_idx: int,
        oracle_return: float,
    ) -> Task:
        # Oracle position is the sign of the next bar's return.
        # In a more sophisticated v0.3, this becomes a learned oracle or
        # a regime-aware target rather than pure perfect-foresight sign.
        oracle_position = float(np.sign(oracle_return))

        last_close = float(window["close"].iloc[-1])
        last_date = (
            window.index[-1].isoformat()
            if isinstance(window.index[-1], (pd.Timestamp, datetime))
            else str(window.index[-1])
        )

        prompt = (
            f"You are evaluating {self.symbol}. The table below shows the "
            f"last {len(window)} OHLCV bars, ending at {last_date} with a "
            f"close of {last_close:.4f}.\n\n"
            "What position should you take for the NEXT bar? Reply with a "
            "single signed number in [-1, 1] where -1 = fully short, "
            "0 = flat, +1 = fully long. Answer with just the number."
        )

        # Compact CSV-ish rendering so the table is readable to an LLM.
        table = "date,open,high,low,close,volume\n" + "\n".join(
            f"{idx},{r.open:.4f},{r.high:.4f},{r.low:.4f},{r.close:.4f},{int(r.volume)}"
            for idx, r in window.iterrows()
        )

        return Task(
            metadata=TaskMetadata(
                task_id=f"trading-{self.symbol}-{end_idx:06d}",
                domain="finance.trading.single_asset",
                difficulty="hard",
                author="synthetic:SimulatedTradingTaskBuilder",
                created_at=datetime.now(timezone.utc),
                tags=["trading", "ohlcv", self.symbol.lower()],
                language="en",
            ),
            prompt=prompt,
            materials=[
                TaskMaterial(
                    kind="table",
                    content=table,
                    mime_type="text/csv",
                    name=f"{self.symbol}_window.csv",
                ),
            ],
            attempt_protocol="one_shot",
            grader_spec=GraderSpec(
                type="exact_match",
                config={
                    "answer": f"{oracle_position:.1f}",
                    "numeric_tolerance": self.position_tolerance,
                },
            ),
            reference_answer=f"{oracle_position:.1f}",
            rubric=(
                "Reference is the sign of the next bar's realized return. "
                "Numeric tolerance allows partial credit when the model "
                "picks the right side of the market with a reasonable size."
            ),
        )


def score_position(model_position: float, next_return: float) -> float:
    """Score a position decision against a realized return.

    Returns a value in ``[0, 1]``: 1.0 means the model took the right side
    of the market at full size; 0.0 means it took the wrong side at full
    size; a flat position scores 0.5 (no information). Useful as the
    payoff function inside custom graders or LLM-judge rubrics that need
    to score open-ended trading explanations.
    """
    if next_return == 0:
        return 0.5
    # Project model_position onto the sign of next_return, normalized to [-1, 1]
    aligned = model_position * np.sign(next_return)
    # Clip first (in case the model reports >1), then linearly map [-1, 1] -> [0, 1].
    aligned = max(-1.0, min(1.0, float(aligned)))
    return 0.5 * (aligned + 1.0)
