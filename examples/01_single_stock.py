"""Use case: is this one stock predictable next-day?

Loads real daily bars, runs every model and both RL agents walk-forward, and
prints a verdict. Expect the answer to be no.

    python examples/01_single_stock.py AAPL
"""

import sys
import warnings

from marketrl import run_pipeline
from marketrl.data import load_sp500

warnings.filterwarnings("ignore")

symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
prices = load_sp500(symbol)
print(f"{symbol}: {len(prices):,} bars, "
      f"{prices.index[0].date()} to {prices.index[-1].date()}\n")

report = run_pipeline(prices, train_size=500, test_size=126, cost_bps=10.0)
print(report.render())
