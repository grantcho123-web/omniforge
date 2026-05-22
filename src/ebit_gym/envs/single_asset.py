"""Single-asset discrete-position trading environment.

v0 design choices:
- Discrete actions: 0=flat, 1=long, 2=short. Position-sizing comes in v1.
- Observation: flattened window of OHLCV plus current position. Box space so
  stock SB3 ``MlpPolicy`` works out of the box.
- Reward: log-return of the held position over the next bar, net of
  proportional transaction cost charged on position *changes*.
- Episode ends when the data runs out (``terminated``) or when equity falls
  below ``ruin_threshold`` (also ``terminated``). ``truncated`` is reserved
  for future time-limit wrappers.
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ebit_gym.data.sources import OHLCV_COLUMNS

_POSITION_FROM_ACTION = np.array([0.0, 1.0, -1.0], dtype=np.float32)


@dataclass
class TradingConfig:
    window_size: int = 32
    transaction_cost: float = 0.0005  # 5 bps per unit of position change
    slippage: float = 0.0001          # 1 bp applied on the executed price
    ruin_threshold: float = 0.5       # terminate if equity drops below 50% of start


class SingleAssetTradingEnv(gym.Env):
    """Gymnasium env wrapping a single-asset OHLCV stream."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        config: TradingConfig | None = None,
    ) -> None:
        super().__init__()
        missing = set(OHLCV_COLUMNS) - set(data.columns)
        if missing:
            raise ValueError(f"data missing required columns: {sorted(missing)}")
        if len(data) <= (config or TradingConfig()).window_size + 1:
            raise ValueError("data must be longer than window_size + 1")

        self.config = config or TradingConfig()
        self._prices = data[OHLCV_COLUMNS].to_numpy(dtype=np.float32)
        self._close = self._prices[:, OHLCV_COLUMNS.index("close")]

        obs_dim = self.config.window_size * len(OHLCV_COLUMNS) + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # Episode state, populated in reset().
        self._t: int = 0
        self._position: float = 0.0
        self._equity: float = 1.0

    # ------------------------------------------------------------------ Gym API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._t = self.config.window_size
        self._position = 0.0
        self._equity = 1.0
        return self._observe(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        new_position = float(_POSITION_FROM_ACTION[int(action)])
        position_change = abs(new_position - self._position)

        # Price evolution: hold the new position from t to t+1.
        price_t = float(self._close[self._t])
        price_t1 = float(self._close[self._t + 1])
        bar_return = (price_t1 - price_t) / price_t

        pnl = new_position * bar_return
        cost = position_change * (self.config.transaction_cost + self.config.slippage)
        reward = pnl - cost

        self._equity *= 1.0 + reward
        self._position = new_position
        self._t += 1

        terminated = (
            self._t >= len(self._close) - 1
            or self._equity <= self.config.ruin_threshold
        )
        truncated = False
        return self._observe(), float(reward), terminated, truncated, self._info()

    # ------------------------------------------------------------------ helpers

    def _observe(self) -> np.ndarray:
        # Normalize the window by the latest close so the policy sees scale-free
        # features. Volume is normalized separately by its own window mean.
        start = self._t - self.config.window_size
        window = self._prices[start : self._t].copy()
        last_close = self._close[self._t - 1]
        window[:, :4] = window[:, :4] / last_close - 1.0
        vol_mean = window[:, 4].mean()
        if vol_mean > 0:
            window[:, 4] = window[:, 4] / vol_mean - 1.0
        flat = window.reshape(-1).astype(np.float32)
        return np.concatenate([flat, np.array([self._position], dtype=np.float32)])

    def _info(self) -> dict:
        return {
            "t": self._t,
            "position": self._position,
            "equity": self._equity,
            "price": float(self._close[min(self._t, len(self._close) - 1)]),
        }
