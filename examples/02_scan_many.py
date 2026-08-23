"""Use case: I tested 120 stocks and three looked great. Are they real?

This is the question that sinks most quantitative research. Testing many
symbols and reporting the winners is guaranteed to produce winners: at a 5%
threshold, 120 coin flips yield six "significant" results on average.

The scan reports the whole distribution and corrects for the multiplicity, so
a finding has to beat chance rather than merely appear in the tail.

    python examples/02_scan_many.py [n_symbols]
"""

import sys
import warnings

from marketrl.data import sp500_symbols
from marketrl.scan import scan_symbols

warnings.filterwarnings("ignore")

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 60
symbols = sp500_symbols()[:limit]
print(f"scanning {len(symbols)} symbols ...\n")

report = scan_symbols(symbols, source="sp500", model="ridge",
                      train_size=500, test_size=126, cost_bps=10.0, progress=True)
print()
print(report.render())
