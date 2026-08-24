"""Use case: at what trading cost does my strategy stop working?

A strategy that trades 140x a year is spending 1.4% of capital per bp of cost.
Sweeping the cost assumption is the fastest way to see whether an edge is real
or is an artifact of assuming free execution -- and it is the sweep most
backtests skip.
"""

import warnings

import pandas as pd

from marketrl.backtest import backtest_positions, positions_from_predictions, to_simple_returns
from marketrl.data import load_sp500
from marketrl.features import make_dataset
from marketrl.supervised import walk_forward_predict

warnings.filterwarnings("ignore")

prices = load_sp500("AAPL")
features, target, _ = make_dataset(prices)
forward = pd.Series(to_simple_returns(target), index=target.index)

outcome = walk_forward_predict(features, target, "ridge", train_size=500, test_size=126)
positions = positions_from_predictions(outcome.predictions)
forward = forward.loc[positions.index]

print(f"AAPL, {len(positions)} out-of-sample days, "
      f"directional accuracy {outcome.metrics.directional_accuracy:.3f}\n")
print(f"{'cost (bps)':>11}{'return':>10}{'Sharpe':>9}{'costs paid':>12}")
print("-" * 42)
for cost_bps in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0):
    result = backtest_positions(positions, forward, cost_bps=cost_bps)
    print(f"{cost_bps:>11.0f}{result.metrics.total_return:>10.1%}"
          f"{result.metrics.sharpe:>9.2f}{result.costs.sum():>12.3f}")

benchmark = backtest_positions(pd.Series(1.0, index=positions.index), forward, cost_bps=10.0)
print(f"\nbuy-and-hold at 10bps: {benchmark.metrics.total_return:.1%} return, "
      f"Sharpe {benchmark.metrics.sharpe:.2f}")
print("\nIf the edge only exists at 0 bps, it does not exist.")
