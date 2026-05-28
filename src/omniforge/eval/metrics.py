"""Risk-adjusted performance metrics for backtest evaluation.

All functions accept either a 1-D numpy array or a pandas Series. ``returns``
are simple per-bar returns (not log). ``equity`` is a cumulative equity curve.
``periods_per_year`` is used to annualize Sharpe/Sortino — pass 252 for daily
bars, 252*390 for 1-min equity bars, etc.
"""
from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray


def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def total_return(equity: ArrayLike) -> float:
    eq = _as_array(equity)
    return float(eq[-1] / eq[0] - 1.0)


def sharpe(returns: ArrayLike, periods_per_year: int = 252) -> float:
    r = _as_array(returns)
    std = r.std(ddof=1)
    if std == 0 or len(r) < 2:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / std)


def sortino(returns: ArrayLike, periods_per_year: int = 252) -> float:
    r = _as_array(returns)
    downside = r[r < 0]
    if len(downside) < 2:
        return 0.0
    dd_std = downside.std(ddof=1)
    if dd_std == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / dd_std)


def max_drawdown(equity: ArrayLike) -> float:
    eq = _as_array(equity)
    peaks = np.maximum.accumulate(eq)
    drawdowns = eq / peaks - 1.0
    return float(drawdowns.min())


def turnover(positions: ArrayLike) -> float:
    """Average absolute change in position per bar.

    A turnover of 0.0 means the policy never traded; 1.0 means it flipped the
    full position every bar on average.
    """
    p = _as_array(positions)
    if len(p) < 2:
        return 0.0
    return float(np.abs(np.diff(p)).mean())
