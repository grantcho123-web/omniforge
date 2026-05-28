"""End-to-end demo: train PPO on the synthetic single-asset env.

Run with the [train] extra installed:

    pip install -e ".[train]"
    python scripts/train_ppo_demo.py

This is a smoke demo — it proves the env drives a stock SB3 policy and produces
the standard tearsheet metrics. It is intentionally tiny (2048 timesteps).

Expect BOTH the random baseline and the trained PPO to lose money. The
synthetic data is a pure random walk with no exploitable signal, and the env
charges 5bps cost + 1bp slippage on every position change. A policy that
turns over ~90% of its book each bar will compound into ruin within a
thousand bars. That is the env being honest, not a bug: a profitable random
policy on this data would mean costs are silently zero.

To see learning, feed real OHLCV (e.g. ``YFinanceOHLCV``) with persistence,
train for >=100k timesteps, and lower turnover by widening the action set or
using a position-sizing wrapper.
"""
from __future__ import annotations

import numpy as np

from omniforge import SingleAssetTradingEnv
from omniforge.data import SyntheticOHLCV
from omniforge.envs.single_asset import TradingConfig
from omniforge.eval import max_drawdown, sharpe, sortino, total_return, turnover

TRAIN_TIMESTEPS = 2048


def make_env(seed: int) -> SingleAssetTradingEnv:
    data = SyntheticOHLCV(n_bars=2048, seed=seed).load()
    return SingleAssetTradingEnv(data, TradingConfig(window_size=32))


def evaluate(env: SingleAssetTradingEnv, predict) -> dict:
    obs, info = env.reset(seed=123)
    rewards, positions, equity = [], [info["position"]], [info["equity"]]
    while True:
        action, _ = predict(obs)
        obs, reward, terminated, truncated, info = env.step(int(action))
        rewards.append(reward)
        positions.append(info["position"])
        equity.append(info["equity"])
        if terminated or truncated:
            break
    rewards = np.array(rewards)
    return {
        "total_return": total_return(equity),
        "sharpe": sharpe(rewards, periods_per_year=252 * 390),
        "sortino": sortino(rewards, periods_per_year=252 * 390),
        "max_drawdown": max_drawdown(equity),
        "turnover": turnover(positions),
        "n_steps": len(rewards),
    }


def main() -> None:
    from stable_baselines3 import PPO

    env = make_env(seed=0)

    print(f"training PPO for {TRAIN_TIMESTEPS} timesteps...")
    model = PPO("MlpPolicy", env, verbose=0, seed=0, n_steps=512)
    model.learn(total_timesteps=TRAIN_TIMESTEPS)

    eval_env = make_env(seed=1)  # held-out seed

    print("\n--- random policy (baseline) ---")
    random_metrics = evaluate(eval_env, lambda obs: (eval_env.action_space.sample(), None))
    for k, v in random_metrics.items():
        print(f"  {k:>14}: {v:.4f}" if isinstance(v, float) else f"  {k:>14}: {v}")

    print("\n--- trained PPO ---")
    ppo_metrics = evaluate(eval_env, model.predict)
    for k, v in ppo_metrics.items():
        print(f"  {k:>14}: {v:.4f}" if isinstance(v, float) else f"  {k:>14}: {v}")


if __name__ == "__main__":
    main()
