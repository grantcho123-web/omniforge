"""Tests for the walk-forward harness."""
from __future__ import annotations

import pandas as pd
import pytest

from omniforge import WalkForward
from omniforge.data import SyntheticOHLCV


def _df(n: int = 1200) -> pd.DataFrame:
    return SyntheticOHLCV(n_bars=n, seed=0).load()


def test_expanding_folds_grow():
    wf = WalkForward(_df(), n_folds=4, mode="expanding", min_train_bars=10)
    folds = wf.folds()
    assert len(folds) == 4
    train_sizes = [len(f.train) for f in folds]
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[0] < train_sizes[-1]
    # All eval segments are the same size in expanding mode.
    eval_sizes = {len(f.eval) for f in folds}
    assert len(eval_sizes) == 1


def test_rolling_folds_have_constant_train_size():
    wf = WalkForward(_df(), n_folds=4, mode="rolling", min_train_bars=10)
    folds = wf.folds()
    train_sizes = {len(f.train) for f in folds}
    assert len(train_sizes) == 1


def test_train_eval_dont_overlap():
    wf = WalkForward(_df(), n_folds=3, min_train_bars=10)
    for fold in wf:
        assert fold.train.index[-1] < fold.eval.index[0]


def test_min_train_bars_skips_short_folds():
    # With only 600 bars and 5 folds (chunk=100), the first fold's train
    # would be 100 bars. min_train_bars=200 should skip it.
    wf = WalkForward(_df(600), n_folds=5, min_train_bars=200)
    folds = wf.folds()
    assert all(len(f.train) >= 200 for f in folds)
    assert len(folds) < 5


def test_rejects_too_short_data():
    with pytest.raises(ValueError, match="too short"):
        list(WalkForward(_df(10), n_folds=5, min_train_bars=0))


def test_rejects_unsorted_index():
    df = _df()
    shuffled = df.sample(frac=1.0, random_state=0)
    with pytest.raises(ValueError, match="sorted"):
        WalkForward(shuffled, n_folds=3)


def test_rejects_invalid_mode():
    with pytest.raises(ValueError, match="mode must be"):
        WalkForward(_df(), n_folds=3, mode="bogus")  # type: ignore[arg-type]
