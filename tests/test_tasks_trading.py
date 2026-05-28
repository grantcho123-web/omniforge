"""Tests for the SimulatedTradingTaskBuilder.

The trading task is a concrete instance of the v0.2 Task schema. These
tests verify the builder produces well-formed tasks, the oracle scoring
is correct, and the integration with the v0.1 OHLCV machinery still works.
"""
from __future__ import annotations

import pandas as pd
import pytest

# Importing ebit_gym.graders registers the reference graders so make_grader works.
import ebit_gym.graders  # noqa: F401
from ebit_gym.core.runner import AttemptRunner
from ebit_gym.data.sources import SyntheticOHLCV
from ebit_gym.models.mock import MockAdapter
from ebit_gym.tasks.trading import SimulatedTradingTaskBuilder, score_position


def _data(n: int = 200, seed: int = 0) -> pd.DataFrame:
    return SyntheticOHLCV(n_bars=n, seed=seed).load()


# ---------------------------------------------------------------- builder


def test_builder_yields_tasks_with_correct_count():
    builder = SimulatedTradingTaskBuilder(
        data=_data(100), window_size=16, stride=4, symbol="SYN"
    )
    tasks = builder.tasks()
    # n - window_size - 1 candidate end indices, then strided
    assert len(tasks) > 0
    # Each task has unique id
    ids = [t.metadata.task_id for t in tasks]
    assert len(set(ids)) == len(ids)


def test_builder_respects_max_tasks():
    builder = SimulatedTradingTaskBuilder(
        data=_data(200), window_size=16, max_tasks=5
    )
    assert len(builder.tasks()) == 5


def test_builder_emits_well_formed_tasks():
    builder = SimulatedTradingTaskBuilder(
        data=_data(100), window_size=16, max_tasks=1, symbol="SPY"
    )
    [task] = builder.tasks()

    assert task.metadata.domain == "finance.trading.single_asset"
    assert "spy" in task.metadata.tags
    assert task.attempt_protocol == "one_shot"
    assert task.grader_spec.type == "exact_match"
    # Reference answer is in {-1.0, 0.0, 1.0}
    assert task.reference_answer in {"-1.0", "0.0", "1.0"}
    # Materials contain a CSV window with the right column headers
    assert len(task.materials) == 1
    mat = task.materials[0]
    assert mat.kind == "table"
    assert mat.mime_type == "text/csv"
    assert mat.content.startswith("date,open,high,low,close,volume\n")
    # Prompt mentions the symbol and the [-1, 1] action space
    assert "SPY" in task.prompt
    assert "[-1, 1]" in task.prompt


def test_builder_oracle_matches_next_bar_sign():
    """The reference position must equal sign(next_return) for every emitted task."""
    df = _data(80)
    close = df["close"].to_numpy()

    builder = SimulatedTradingTaskBuilder(data=df, window_size=8, stride=1, max_tasks=20)
    for task in builder:
        # task_id encodes the end index: trading-{SYMBOL}-{end_idx:06d}
        end_idx = int(task.metadata.task_id.split("-")[-1])
        expected_sign = (close[end_idx] - close[end_idx - 1]) / close[end_idx - 1]
        if expected_sign > 0:
            assert task.reference_answer == "1.0"
        elif expected_sign < 0:
            assert task.reference_answer == "-1.0"
        else:
            assert task.reference_answer == "0.0"


def test_builder_rejects_missing_columns():
    bad = pd.DataFrame({"open": [1, 2], "close": [1, 2]})  # missing high/low/volume
    with pytest.raises(ValueError, match="missing required columns"):
        SimulatedTradingTaskBuilder(data=bad, window_size=2)


def test_builder_rejects_too_short_data():
    with pytest.raises(ValueError, match="longer than window_size"):
        SimulatedTradingTaskBuilder(data=_data(10), window_size=20)


def test_builder_rejects_bad_window_size():
    with pytest.raises(ValueError, match="window_size"):
        SimulatedTradingTaskBuilder(data=_data(100), window_size=1)


def test_builder_rejects_bad_stride():
    with pytest.raises(ValueError, match="stride"):
        SimulatedTradingTaskBuilder(data=_data(100), window_size=16, stride=0)


def test_builder_rejects_bad_tolerance():
    with pytest.raises(ValueError, match="position_tolerance"):
        SimulatedTradingTaskBuilder(
            data=_data(100), window_size=16, position_tolerance=-0.1
        )


# --------------------------------------------------------- score_position


def test_score_position_perfect_long():
    assert score_position(1.0, 0.01) == 1.0


def test_score_position_perfect_short():
    assert score_position(-1.0, -0.01) == 1.0


def test_score_position_wrong_side():
    assert score_position(1.0, -0.01) == 0.0
    assert score_position(-1.0, 0.01) == 0.0


def test_score_position_flat_returns_half():
    assert score_position(0.0, 0.01) == 0.5
    assert score_position(0.0, -0.01) == 0.5


def test_score_position_zero_return_is_indeterminate():
    assert score_position(0.5, 0.0) == 0.5


def test_score_position_clips_oversize():
    """A model that returns 5.0 should be treated as 1.0 (full long)."""
    assert score_position(5.0, 0.01) == 1.0
    assert score_position(-5.0, -0.01) == 1.0


def test_score_position_partial_credit_for_correct_side_smaller_size():
    s = score_position(0.5, 0.01)
    assert 0.5 < s < 1.0


# ----------------------------------------------------- end-to-end with runner


def test_trading_task_runs_through_attempt_runner():
    """Smoke test: the trading task is a citizen of the v0.2 platform —
    AttemptRunner drives it and the grader scores it like any other task."""
    builder = SimulatedTradingTaskBuilder(
        data=_data(100), window_size=16, max_tasks=2, symbol="SYN"
    )
    [task1, task2] = builder.tasks()

    # Mock model "answers" each task with the correct reference position.
    model = MockAdapter(
        responses=[task1.reference_answer, task2.reference_answer], name="mock:trader"
    )
    runner = AttemptRunner(model)

    for task in (task1, task2):
        attempt = runner.run(task)
        from ebit_gym.core.grader import make_grader

        grader = make_grader(task.grader_spec)
        result = grader.grade(task, attempt)
        # Perfect oracle response should match (within numeric tolerance)
        assert result.passed
        assert result.score == 1.0
