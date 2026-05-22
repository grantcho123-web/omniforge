# ebit-gym

[![ci](https://github.com/grantcho123-web/ebit-gym/actions/workflows/ci.yml/badge.svg)](https://github.com/grantcho123-web/ebit-gym/actions/workflows/ci.yml)

Reinforcement-learning environments for financial analysis.

`ebit-gym` is an [ebit](https://ebitglobal.ai) product offering. The v0
ships a single-asset trading environment that conforms to the
[Gymnasium](https://gymnasium.farama.org) API and works with any standard RL
library (Stable-Baselines3, CleanRL, RLlib).

## Status

Pre-alpha (v0.0.1). API will break.

## Install

```bash
# Once published:
pip install ebit-gym

# From source (recommended during development):
pip install -e ".[dev,train]"
```

`uv` is recommended for environment management:

```bash
uv venv -p 3.11
source .venv/bin/activate
uv pip install -e ".[dev,train]"
```

## Quickstart

```python
from ebit_gym import SingleAssetTradingEnv
from ebit_gym.data import SyntheticOHLCV

data = SyntheticOHLCV(n_bars=2048, seed=0).load()
env = SingleAssetTradingEnv(data)

obs, info = env.reset(seed=0)
done = False
while not done:
    action = env.action_space.sample()  # 0=flat, 1=long, 2=short
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"final equity: {info['equity']:.4f}")
```

### Train PPO end-to-end

```bash
pip install -e ".[train]"
python scripts/train_ppo_demo.py
```

This runs a 2048-timestep PPO training pass and prints a tearsheet (total
return, Sharpe, Sortino, max drawdown, turnover) against a random baseline.
Both lose money on the default synthetic data — that's the env being honest
about costs, not a bug. See the script docstring for why.

### Real data

```python
from ebit_gym.data import YFinanceOHLCV

data = YFinanceOHLCV("SPY", start="2015-01-01", end="2024-12-31").load()
```

Requires the `[data]` extra.

### Walk-forward backtest on SPY

```bash
pip install -e ".[data,train]"
python scripts/spy_tearsheet.py
```

Trains a continuous-action PPO on an expanding walk-forward split of SPY
2015-2024 (5 folds, daily) and prints per-fold and aggregate out-of-sample
metrics against a buy-and-hold baseline:

```
=== aggregate out-of-sample ===
                     PPO             B&H
    total_return        -0.4198        +1.7652
          sharpe        -0.4497        +0.8135
         sortino        -0.6047        +0.9623
          max_dd        -0.4569        -0.3382
```

PPO loses to B&H — this is expected and honest. With only ~10k training
steps per fold, 5 raw OHLCV features, and a shallow MLP, there is nowhere
near enough capacity to extract a tradeable signal against a 10-year bull
market. What the script *does* prove:

- No train/eval leakage (walk-forward is correct)
- Transaction costs are charged on every position change
- Metrics are computed against a held-out segment, not training data
- The same harness will surface a real signal once you bring richer
  features, a longer training budget, and a model with capacity to match.

That distinction — "honest harness with weak baseline" vs "rigged harness
with cherry-picked Sharpe" — is the value of using `ebit-gym` over a
hand-rolled backtest.

## Design

- **Env API:** Gymnasium-compatible (`reset`, `step`, `observation_space`, `action_space`).
- **Observations:** windowed OHLCV (close-normalized) plus current position, flattened to `Box`.
- **Actions:** `Discrete(3)` → `{flat, long, short}`. Continuous position sizing lands in v1.
- **Reward:** position-weighted bar return, net of proportional transaction cost and slippage.
- **Frictions:** transaction cost and slippage are baked in by default — silent zero-cost backtests are the most common bug in financial RL.
- **Reproducibility:** seeded RNG, deterministic synthetic data, no hidden time-leakage.

## Roadmap

- **v0** (current): single-asset, discrete actions, synthetic + yfinance data, smoke tests.
- **v1:** continuous position sizing, multi-asset (portfolio), walk-forward backtest harness, CLI, docs site.
- **v2:** hosted platform — bring-your-own data, managed training runs, signal/policy export.

## Layout

```
src/ebit_gym/
  envs/          Gymnasium environments
  data/          OHLCV data sources (synthetic in core; yfinance under [data])
  eval/          Risk-adjusted metrics (Sharpe, Sortino, max DD, turnover)
  wrappers/      Reserved for env wrappers (normalization, time-limits, ...)
tests/           pytest smoke tests — run with `pytest`
```

## License

Apache-2.0.
