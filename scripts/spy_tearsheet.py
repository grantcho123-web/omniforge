"""SPY walk-forward tearsheet.

Pulls SPY daily OHLCV (2015-2024), runs an expanding walk-forward backtest
with continuous PPO, and prints per-fold + aggregate out-of-sample metrics.

Run with the [data, train] extras installed:

    pip install -e ".[data,train]"
    python scripts/spy_tearsheet.py

This is a *credibility* demo, not a profitable strategy. The point is to show
the harness produces honest, methodologically-correct numbers (no train/eval
leakage, costs charged, real data). With only price-derived features and a
shallow MLP, do not expect to beat buy-and-hold.
"""
from __future__ import annotations

import argparse

import numpy as np

from omniforge import SingleAssetTradingEnv, WalkForward
from omniforge.data import YFinanceOHLCV
from omniforge.envs.single_asset import TradingConfig
from omniforge.eval import max_drawdown, sharpe, sortino, total_return, turnover

PERIODS_PER_YEAR_DAILY = 252


def eval_policy(env: SingleAssetTradingEnv, predict) -> dict:
    obs, info = env.reset(seed=0)
    rewards, positions, equity = [], [info["position"]], [info["equity"]]
    while True:
        action, _ = predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        positions.append(info["position"])
        equity.append(info["equity"])
        if terminated or truncated:
            break
    rewards_arr = np.array(rewards)
    return {
        "total_return": total_return(equity),
        "sharpe": sharpe(rewards_arr, periods_per_year=PERIODS_PER_YEAR_DAILY),
        "sortino": sortino(rewards_arr, periods_per_year=PERIODS_PER_YEAR_DAILY),
        "max_dd": max_drawdown(equity),
        "turnover": turnover(positions),
        "n_steps": len(rewards_arr),
        "rewards": rewards_arr,
    }


def buy_and_hold(env: SingleAssetTradingEnv) -> dict:
    def predict(_obs):
        return np.array([1.0], dtype=np.float32), None
    return eval_policy(env, predict)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--timesteps-per-fold", type=int, default=10_000)
    parser.add_argument("--window", type=int, default=32)
    args = parser.parse_args()

    from stable_baselines3 import PPO

    print(f"fetching {args.symbol} {args.start}..{args.end}")
    data = YFinanceOHLCV(args.symbol, start=args.start, end=args.end, interval="1d").load()
    print(f"  → {len(data)} bars\n")

    config = TradingConfig(window_size=args.window)
    walk = WalkForward(data, n_folds=args.n_folds, mode="expanding", min_train_bars=256)

    fold_rows = []
    bh_rows = []
    all_rewards: list[np.ndarray] = []
    all_bh_rewards: list[np.ndarray] = []

    for fold in walk:
        tr0, tr1 = fold.train.index[0].date(), fold.train.index[-1].date()
        ev0, ev1 = fold.eval.index[0].date(), fold.eval.index[-1].date()
        print(
            f"fold {fold.fold}: train {tr0}→{tr1} ({len(fold.train)} bars), "
            f"eval {ev0}→{ev1} ({len(fold.eval)} bars)"
        )
        train_env = SingleAssetTradingEnv(fold.train, config, positions=None)
        model = PPO("MlpPolicy", train_env, verbose=0, seed=fold.fold, n_steps=512)
        model.learn(total_timesteps=args.timesteps_per_fold)

        eval_env = SingleAssetTradingEnv(fold.eval, config, positions=None)
        ppo_metrics = eval_policy(eval_env, model.predict)
        bh_env = SingleAssetTradingEnv(fold.eval, config, positions=None)
        bh_metrics = buy_and_hold(bh_env)

        all_rewards.append(ppo_metrics.pop("rewards"))
        all_bh_rewards.append(bh_metrics.pop("rewards"))
        fold_rows.append(ppo_metrics)
        bh_rows.append(bh_metrics)

        print(
            f"  PPO  → ret {ppo_metrics['total_return']:+.2%}  "
            f"sharpe {ppo_metrics['sharpe']:+.2f}  "
            f"DD {ppo_metrics['max_dd']:+.2%}  "
            f"turnover {ppo_metrics['turnover']:.2f}"
        )
        print(
            f"  B&H  → ret {bh_metrics['total_return']:+.2%}  "
            f"sharpe {bh_metrics['sharpe']:+.2f}  "
            f"DD {bh_metrics['max_dd']:+.2%}"
        )

    print("\n=== aggregate out-of-sample ===")
    agg_rewards = np.concatenate(all_rewards)
    agg_bh = np.concatenate(all_bh_rewards)
    agg_equity = np.cumprod(1.0 + agg_rewards)
    agg_bh_equity = np.cumprod(1.0 + agg_bh)
    py = PERIODS_PER_YEAR_DAILY
    rows = [
        ("total_return", total_return(agg_equity), total_return(agg_bh_equity)),
        ("sharpe", sharpe(agg_rewards, py), sharpe(agg_bh, py)),
        ("sortino", sortino(agg_rewards, py), sortino(agg_bh, py)),
        ("max_dd", max_drawdown(agg_equity), max_drawdown(agg_bh_equity)),
    ]
    print(f"            {'PPO':>12}    {'B&H':>12}")
    for label, ppo_v, bh_v in rows:
        print(f"  {label:>14}    {ppo_v:>+11.4f}    {bh_v:>+11.4f}")


if __name__ == "__main__":
    main()
