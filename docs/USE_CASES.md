# Use cases

Five things this package is actually for. Every number below is real output on
real market data (S&P 500 daily bars, 2013–2018), reproducible with the command
shown.

---

## 1. "Is this stock predictable next-day?"

```bash
python examples/01_single_stock.py AAPL
# or: marketrl run --symbol AAPL --source sp500
```

Runs three supervised models, two RL agents and buy-and-hold over the same
out-of-sample window, all net of costs.

**Real result — AAPL, 506 out-of-sample days:**

```
  model                 dir.acc  p-value      IC      RMSE
  persistence             0.490   0.6876   0.035   0.01701
  ridge                   0.476   0.8668  -0.014   0.01312
  random_forest           0.500   0.5177  -0.044   0.01263
  gradient_boosting       0.462   0.9585  -0.045   0.01326

  strategy               return    CAGR  Sharpe   maxDD  turnover
  buy_and_hold            65.0%   28.3%    1.35   19.4%      0.5x
  reinforce                4.2%    2.1%    0.20   39.2%     83.2x
  random_forest            2.4%    1.2%    0.16   35.5%     81.2x
  ridge                  -26.8%  -14.4%   -0.68   31.6%    113.1x
  q_learning             -35.7%  -19.7%   -1.10   40.5%    193.2x
  gradient_boosting      -53.6%  -31.7%   -1.81   56.9%    164.8x
```

No model beats a coin flip; nothing beats holding the stock. **That is the
answer**, and getting it reliably is the point of the tool.

---

## 2. "I scanned 120 stocks and found three winners. Are they real?"

This is the question that sinks most quantitative research, and the one worth
the most money. Testing many symbols and reporting the winners is *guaranteed*
to produce winners.

```bash
marketrl scan --limit 120 --model ridge
# or: python examples/02_scan_many.py 120
```

**Real result — 116 S&P 500 stocks, 506 out-of-sample days each:**

```
DIRECTIONAL ACCURACY ACROSS SYMBOLS
  mean 0.4974   median 0.4960   sd 0.0239
  range 0.425 to 0.553
  sd predicted by chance alone 0.0222   (observed/predicted = 1.08x)

MULTIPLE COMPARISONS
  nominally significant (p < 0.05)   5 of 116
  expected from chance alone                5.8
  survive Benjamini-Hochberg FDR            0
  survive Bonferroni                        0

VERSUS BUY-AND-HOLD (net of costs)
  beat benchmark on Sharpe                  9 of 116 (8%)
  mean excess Sharpe                        -1.450
  median annual turnover                    139x
```

Read the third line of the first block carefully. The spread in accuracy across
116 stocks is **1.08× what pure chance predicts** — meaning the difference
between the "best" stock (55.3%) and the "worst" (42.5%) is noise, top to
bottom. There is no per-symbol skill to rank, and five nominal hits against 5.8
expected is not a discovery.

Without this correction you would report AMAT at 55.3%, p = 0.009, and be
completely wrong.

---

## 3. "At what trading cost does my edge disappear?"

```bash
python examples/03_cost_sensitivity.py
```

**Real result — AAPL ridge strategy:**

| cost (bps) | return | Sharpe |
|---:|---:|---:|
| 0 | −8.2% | −0.11 |
| 1 | −10.2% | −0.17 |
| 5 | −18.0% | −0.39 |
| 10 | −26.8% | −0.68 |
| 20 | −41.7% | −1.24 |

A strategy trading 113× a year spends about 1.1% of capital per basis point of
cost. If an edge only exists at 0 bps, it does not exist. This sweep takes
seconds and is the single most informative thing you can do to a promising
backtest.

---

## 4. "Do the RL agents actually work, or is my null result a bug?"

The most important question about any negative finding. Answer it by running
the same agents on a market that *is* solvable.

```bash
python examples/04_rl_agent.py
```

**Real result:**

```
A. A SOLVABLE MARKET -- the agents should nearly max it out
  q_learning   earned  5.127 of 6.000 possible (85.5%)   episode reward +0.047 -> +1.206
  reinforce    earned  5.916 of 6.000 possible (98.6%)   episode reward +0.784 -> +5.619

B. REAL PRICES -- same agents, walk-forward, net of costs
  q_learning         return  -35.7%  Sharpe -1.10  turnover 193.2x
  reinforce          return    4.2%  Sharpe  0.20  turnover  83.2x
  buy_and_hold       return   65.0%  Sharpe  1.35  turnover   0.5x
```

The agents capture 85% and 99% of the theoretical maximum when there is
something to capture, with clearly rising learning curves. On real prices they
earn nothing. **The implementation works; the market is the problem.**

`marketrl selftest` automates this check end-to-end.

---

## 5. "Am I leaking future information into my features?"

The most expensive bug in quantitative finance, and invisible in any metric —
a leak makes results *better*, so nothing alerts you.

```python
from marketrl import assert_no_lookahead, load_sp500

assert_no_lookahead(load_sp500("AAPL"))   # raises if any feature reads ahead
```

It rewrites every bar after a cut point with random noise and asserts that no
feature value *at or before* the cut changed. Anything that moved was reading
the future.

Add it to your own feature pipeline and to CI (this repo runs it in the
`validation` job on every commit). The test suite here plants a
deliberate leak (`shift(-1)` on the price) to confirm the check catches it —
because a safety check nobody has ever seen fail is not a safety check.

---

## Reproducing the published numbers

The full per-symbol output behind the scan tables is committed as
[`results-scan-ridge.json`](results-scan-ridge.json) and
[`results-scan-gradient-boosting.json`](results-scan-gradient-boosting.json),
so the aggregates can be checked without re-running anything:

```bash
python -c "import json; d=json.load(open('docs/results-scan-ridge.json')); \
  print(d['n_symbols'], d['mean_directional_accuracy'], d['fdr_discoveries'])"
```

Regenerate them with:

```bash
marketrl scan --limit 120 --model ridge --json > docs/results-scan-ridge.json
```

The dataset is frozen, so these are reproducible exactly.

## Which knobs actually matter

| Setting | Effect |
|---|---|
| `--cost-bps` | Dominates everything at 130–200× turnover. Start at 10, never 0. |
| `--threshold` | A dead zone around a weak forecast. The main lever for cutting turnover. |
| `--train` / `--test` | Larger train = more data, fewer folds. Small test blocks make noisy per-fold metrics. |
| `--min-test-days` | Guards the scan. At 28 test days a coin flip clears p < 0.05 routinely and posts Sharpe > 3. |
| `--long-only` | Removes shorting. More realistic for most accounts; usually raises returns in a bull sample. |

## Two failure modes to watch for in your own use

**The sample is one regime.** The bundled dataset is 2013–2018, a period with
no sustained bear market. Nothing here has been tested through a crash.

**Re-running until something wins.** Every re-run with new settings is another
comparison, and none of the p-values above are corrected for how many times
*you* have tried. The verdict text says this on purpose.
