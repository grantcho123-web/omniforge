"""Walk-forward backtesting.

Standard methodology for time-series RL: train on a rolling or expanding
window, evaluate on the immediately following out-of-sample segment, then
slide forward. Aggregates the held-out segments into a single Sharpe / DD
/ tearsheet so the reported number actually corresponds to deployable
performance — not to a model that has seen its eval data.

Two modes:
- ``mode="expanding"`` (default): train window grows each fold, eval stays
  the same size. Best when you believe more history helps.
- ``mode="rolling"``: train and eval windows both slide forward at fixed
  size. Best when regimes change and old data is misleading.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import pandas as pd

Mode = Literal["expanding", "rolling"]


@dataclass(frozen=True)
class WalkForwardFold:
    """One (train, eval) split."""

    fold: int
    train: pd.DataFrame
    eval: pd.DataFrame


@dataclass
class WalkForward:
    """Generates ordered (train, eval) folds over a time-indexed DataFrame.

    Args:
        data: OHLCV (or any time-indexed) DataFrame, must be sorted.
        n_folds: number of eval segments. The data is split into
            ``n_folds + 1`` equal-size chunks; the first chunk seeds the
            initial train set, then each subsequent chunk becomes one eval
            fold (with the running history as train).
        mode: "expanding" or "rolling".
        min_train_bars: skip folds where the training set would be shorter
            than this. Protects against degenerate first folds.
    """

    data: pd.DataFrame
    n_folds: int = 5
    mode: Mode = "expanding"
    min_train_bars: int = 256

    def __post_init__(self) -> None:
        if self.n_folds < 1:
            raise ValueError("n_folds must be >= 1")
        if self.mode not in ("expanding", "rolling"):
            raise ValueError(f"mode must be 'expanding' or 'rolling', got {self.mode!r}")
        if not self.data.index.is_monotonic_increasing:
            raise ValueError("data index must be sorted ascending")

    def __iter__(self) -> Iterator[WalkForwardFold]:
        n = len(self.data)
        chunk = n // (self.n_folds + 1)
        if chunk < 2:
            raise ValueError(
                f"data too short for n_folds={self.n_folds}: each chunk would be {chunk} bars"
            )

        for fold in range(self.n_folds):
            eval_start = (fold + 1) * chunk
            eval_end = eval_start + chunk
            train_start = 0 if self.mode == "expanding" else eval_start - chunk
            train = self.data.iloc[train_start:eval_start]
            eval_ = self.data.iloc[eval_start:eval_end]
            if len(train) < self.min_train_bars:
                continue
            yield WalkForwardFold(fold=fold, train=train, eval=eval_)

    def folds(self) -> list[WalkForwardFold]:
        """Eager list of folds; convenient for iteration with progress bars."""
        return list(self)
