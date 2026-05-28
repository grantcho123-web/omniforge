"""Single-asset trading environment.

Design:
- Actions: either ``Discrete(len(positions))`` mapped through a custom
  position grid (e.g. ``[-1, -0.5, 0, 0.5, 1]``), or ``Box([-1, 1])`` for
  continuous position sizing. Selected via the ``positions`` argument.
- Observation: flattened window of OHLCV (close-normalized) plus current
  position. ``Box`` space so stock ``MlpPolicy`` works out of the box.
- Reward: position-weighted next-bar return net of proportional transaction
  cost + slippage, charged on position *changes*.
- Episode ends when the data runs out or equity falls below
  ``ruin_threshold``. ``truncated`` is reserved for future time-limit wrappers.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from omniforge.data.sources import OHLCV_COLUMNS

# Default discrete grid: flat / long / short. Kept for the v0 API contract.
DEFAULT_POSITIONS: tuple[float, ...] = (0.0, 1.0, -1.0)


@dataclass
class TradingConfig:
    window_size: int = 32
    transaction_cost: float = 0.0005  # 5 bps per unit of position change
    slippage: float = 0.0001          # 1 bp applied on the executed price
    ruin_threshold: float = 0.5       # terminate if equity drops below 50% of start


class SingleAssetTradingEnv(gym.Env):
    """Gymnasium env wrapping a single-asset OHLCV stream.

    Args:
        data: DataFrame with at least the columns ``[open, high, low, close, volume]``.
        config: env hyperparameters; defaults are conservative.
        positions: action-space spec.
            - ``None`` → continuous ``Box([-1, 1])``. Action is the target position.
            - ``Sequence[float]`` → ``Discrete(len(positions))``; action indexes
              into the grid. Pass e.g. ``[-1, -0.5, 0, 0.5, 1]`` for sized
              positions, or ``(0, 1, -1)`` for the v0 flat/long/short default.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        config: TradingConfig | None = None,
        positions: Sequence[float] | None = DEFAULT_POSITIONS,
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

        if positions is None:
            self._positions = None
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        else:
            grid = np.asarray(list(positions), dtype=np.float32)
            if grid.size == 0:
                raise ValueError("positions must be non-empty or None for continuous")
            if (np.abs(grid) > 1.0).any():
                raise ValueError("positions must lie in [-1, 1]")
            self._positions = grid
            self.action_space = spaces.Discrete(len(grid))

        obs_dim = self.config.window_size * len(OHLCV_COLUMNS) + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

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

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        new_position = self._position_from_action(action)
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

    def _position_from_action(self, action) -> float:
        if self._positions is None:
            # Continuous: action is a length-1 array (or scalar). Clip to [-1, 1].
            value = float(np.asarray(action).reshape(-1)[0])
            return float(np.clip(value, -1.0, 1.0))
        return float(self._positions[int(action)])

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
