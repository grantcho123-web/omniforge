"""Data sources for OHLCV bars.

The data layer is intentionally thin: anything that returns a pandas DataFrame
with columns ``[open, high, low, close, volume]`` indexed by timestamp is a
valid source. ``SyntheticOHLCV`` ships in core for deterministic tests and
demos that work without network access. Real-market sources (yfinance, etc.)
live behind the ``data`` extras.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass
class SyntheticOHLCV:
    """Deterministic geometric-Brownian-motion OHLCV generator.

    Used as the default fixture so tests and quickstart demos do not require
    network access or a paid data vendor.
    """

    n_bars: int = 1024
    start_price: float = 100.0
    drift: float = 0.0001
    volatility: float = 0.01
    seed: int = 0

    def load(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        returns = rng.normal(self.drift, self.volatility, size=self.n_bars)
        close = self.start_price * np.exp(np.cumsum(returns))

        # Build OHLC around the close path with a small intrabar range.
        intrabar = np.abs(rng.normal(0.0, self.volatility / 2, size=self.n_bars)) * close
        open_ = np.concatenate([[self.start_price], close[:-1]])
        high = np.maximum(open_, close) + intrabar
        low = np.minimum(open_, close) - intrabar
        volume = rng.integers(1_000, 10_000, size=self.n_bars).astype(float)

        index = pd.date_range("2020-01-01", periods=self.n_bars, freq="1min")
        return pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        )
