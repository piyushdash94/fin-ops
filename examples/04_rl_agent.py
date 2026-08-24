"""Use case: train an RL agent and watch it actually learn.

Trains on a market that IS solvable (feature 0 reveals tomorrow's sign) so the
learning curve is unambiguous, then on real data where it is not. The contrast
is the point: the same agent that recovers most of the available profit in a
solvable market earns nothing on real prices.
"""

import warnings

import numpy as np
import pandas as pd

from marketrl.agents import QLearningAgent, ReinforceAgent
from marketrl.backtest import to_simple_returns, walk_forward_rl
from marketrl.data import load_sp500
from marketrl.env import TradingEnv
from marketrl.features import make_dataset

warnings.filterwarnings("ignore")

print("=" * 62)
print("A. A SOLVABLE MARKET -- the agents should nearly max it out")
print("=" * 62)
rng = np.random.default_rng(0)
n = 600
signal = rng.standard_normal(n)
forward = np.sign(signal) * 0.01
features = np.column_stack([signal, rng.standard_normal(n)])
env = TradingEnv(features, forward, cost_bps=1.0)
best_possible = float(np.abs(forward).sum())

for agent in (QLearningAgent(episodes=60), ReinforceAgent(episodes=150)):
    agent.fit(env)
    earned = env.run(agent)["net_returns"].sum()
    curve = np.array(agent.rewards_)
    print(f"  {agent.name:<12} earned {earned:6.3f} of {best_possible:.3f} possible "
          f"({earned / best_possible:5.1%})   "
          f"episode reward {curve[:10].mean():+.3f} -> {curve[-10:].mean():+.3f}")

print()
print("=" * 62)
print("B. REAL PRICES -- same agents, walk-forward, net of costs")
print("=" * 62)
prices = load_sp500("AAPL")
features_df, target, _ = make_dataset(prices)
forward_real = pd.Series(to_simple_returns(target), index=target.index)

for name, factory in (
    ("q_learning", lambda: QLearningAgent(episodes=60)),
    ("reinforce", lambda: ReinforceAgent(episodes=150)),
):
    result = walk_forward_rl(features_df, forward_real, factory, name=name,
                             train_size=500, test_size=126, cost_bps=10.0)
    print("  " + result.summary())

benchmark_index = result.positions.index
from marketrl.backtest import backtest_positions  # noqa: E402

benchmark = backtest_positions(
    pd.Series(1.0, index=benchmark_index), forward_real.loc[benchmark_index],
    cost_bps=10.0, name="buy_and_hold",
)
print("  " + benchmark.summary())
print("\nThe agents learn. There is just nothing here to learn.")
