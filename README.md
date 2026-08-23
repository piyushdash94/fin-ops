# fin-ops

Two toolkits about the same discipline — measuring what something really costs,
and refusing to believe a number that hasn't been tested.

| Package | What it does |
|---|---|
| **[`marketrl`](#marketrl)** | Next-day stock return prediction with ML and RL, evaluated walk-forward and net of costs |
| **[`finops`](docs/finops.md)** | Token accounting and cost control for Claude-powered applications |

```bash
pip install -e ".[dev]"
python -m pytest        # 187 tests
```

---

# marketrl

Predicts the next day's return for a stock using supervised ML and two
reinforcement-learning agents, then tells you — usually — that it didn't work.

That last part is the feature. Building a model that appears to predict the
market is easy and takes about thirty lines; nearly all of them are wrong for
one of four reasons, and this package is organized around those four.

### The four ways a price model lies to you

**1. Lookahead.** A centered rolling window, a scaler fitted on the whole
series, `shift(-1)` in the wrong place — any of these produce a beautiful
backtest and a losing strategy. Rather than trusting review,
`assert_no_lookahead()` mechanically rewrites the future with noise and asserts
that no feature computed at time *t* changed. It runs in the test suite, and a
test deliberately plants a leak to confirm the check catches it.

**2. Predicting the price instead of the return.** Predicting tomorrow's *price*
scores R² ≈ 0.999 and means nothing — tomorrow's price is roughly today's price.
The target here is next-day **log return**, the part that is genuinely unknown.

**3. Ignoring costs.** Both the backtester and the RL reward charge
transaction costs on every change in exposure. This matters enormously: the
strategies below trade 130–220× a year, so at 10bps a side that is a 13–22%
annual drag. Costs are inside the RL *reward*, not subtracted afterwards, so the
agent's own objective includes the friction it creates.

**4. Mistaking noise for signal.** 52% directional accuracy sounds like an edge.
Over 1,400 days it is a coin flip (p = 0.27). Every directional number in the
report ships with the probability of seeing it by chance.

### Real results on real data

The headline experiment: one model, 116 real S&P 500 stocks, 506 out-of-sample
days each, net of 10bps costs.

```console
$ marketrl scan --limit 120 --model ridge

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

VERDICT
  5 symbols cleared p < 0.05 individually, but chance alone predicts 5.8.
  None survive correction for having tested 116 of them. There is no
  evidence of next-day predictability here.
```

The third line of the first block is the finding. The spread in accuracy across
116 stocks is **1.08x what pure chance predicts** — so the gap between the best
stock (55.3%) and the worst (42.5%) is noise from end to end. There is no
per-symbol skill to rank. Five nominal hits against 5.8 expected is not a
discovery, and none survive correction.

Report the winner without that correction and you would publish AMAT at 55.3%,
p = 0.009, and be entirely wrong.

**It replicates across model families.** Re-running the identical scan with
gradient boosting — a completely different inductive bias — reaches the same
conclusion:

| | ridge | gradient boosting |
|---|---:|---:|
| mean directional accuracy | 0.4974 | 0.5002 |
| observed spread ÷ chance spread | 1.08× | 0.95× |
| nominally significant (of 116) | 5 | 5 |
| expected by chance | 5.8 | 5.8 |
| survive FDR correction | **0** | **0** |
| beat buy-and-hold | 9 (8%) | 7 (6%) |
| median annual turnover | 139× | 174× |

Two unrelated model families, both landing exactly on chance. That is what an
efficient market looks like from the inside.

### Why you should believe the null result

A negative finding is only worth anything if the tool could have found a
positive one. Three checks establish that, and all three run in CI:

**The agents solve a solvable market.** Given a synthetic market where one
feature reveals tomorrow's sign, Q-learning captures 85.5% of the theoretical
maximum profit and REINFORCE 98.6%, both with clearly rising learning curves.
On real prices the same agents earn nothing. The implementation works; the
market is the problem.

**The pipeline detects a planted edge.** `marketrl selftest` runs the whole
thing on a market with known predictability (all models clear p < 0.01,
IC ≈ 0.13) and on a provably efficient one (nothing significant, verdict says
so). A backtester that cannot return a null result will eventually tell you to
trade on noise.

**A deliberate leak is caught.** The test suite plants `shift(-1)` on the price
into the feature set and asserts `assert_no_lookahead()` raises — because a
safety check nobody has watched fail is not a safety check.

## Usage

```console
$ marketrl run  --symbol AAPL --source sp500     # one stock, every model
$ marketrl scan --limit 120 --model ridge        # many stocks, FDR-corrected
$ marketrl data --symbol MSFT --source sp500     # summarize price history
$ marketrl selftest                              # prove the pipeline works
```

Worked examples with real output live in **[docs/USE_CASES.md](docs/USE_CASES.md)**;
the runnable scripts are in [`examples/`](examples/).

```python
from marketrl import load, run_pipeline

print(run_pipeline(load("AAPL"), cost_bps=10.0).render())
```

### Data sources

| Source | What it is |
|---|---|
| `sp500` | A fixed public dataset of **real** daily bars — 505 S&P 500 symbols, 2013-02 to 2018-02. Every result above comes from this. |
| `yahoo` | Yahoo Finance's chart API, used directly (one fewer dependency than `yfinance`, and legible failures). Live and current. |
| `stooq` | A second free source, no key required. |
| a CSV path | Any other dataset you have on disk. |
| `synthetic` | Simulated prices with tunable predictability, for validation. |

The `sp500` dataset is the default benchmark precisely because it is frozen: a
live API changes between runs, so a number quoted from one cannot be reproduced.
Its limits are real, though — a single mostly-rising regime, no dividend
adjustment, and survivorship bias baked in (these are companies that were in the
index in 2018).

**A network failure raises.** It never degrades to simulated prices. Synthetic
data must be asked for by name, and every report built on it carries a banner.

## What's inside

- **26 causal features** — multi-horizon momentum normalized by realized
  volatility, RSI, MACD, Bollinger z-scores, ATR, volume z-scores, range
  position, cyclical calendar encodings. All scale-free, so a model still
  applies after the stock doubles (there's a test for that).
- **Supervised models** — ridge, random forest, histogram gradient boosting,
  all heavily regularized because with ~25 features and a signal-to-noise ratio
  near zero, an unconstrained learner fits noise and loses to predicting zero.
  Baselines (zero, mean, persistence) run alongside and sometimes win.
- **`QLearningAgent`** — tabular Q-learning over a quantile-discretized state,
  fitted on training data only. Small enough to print and read.
- **`ReinforceAgent`** — policy gradient with a value baseline, entropy
  regularization and Adam, in ~80 lines of NumPy. No PyTorch. The entropy bonus
  isn't decoration: without it the policy collapses onto one action within a few
  episodes, because on noisy data any lucky streak looks decisive.
- **`TradingEnv`** — long/flat/short, costs in the reward, current position in
  the state (which is what makes it properly Markovian: what to do next depends
  on what you hold, because changing your mind costs money).
- **Walk-forward everything** — a fresh model *and a fresh agent* per fold, with
  an embargo gap so the last training label isn't known only on the first test
  day. K-fold on price data lets a model train on Thursday to predict Wednesday.

Both agents are verified to solve a market that *is* solvable, recovering >50%
of the theoretical maximum profit with rising learning curves — so a null result
on real data is a statement about the market, not a broken implementation.

## Honest limitations

- **Daily bars only.** No intraday, no order book, no borrow costs for shorts,
  no slippage model beyond a flat per-unit cost.
- **Execution is assumed at the close** of the day the decision is made. Real
  fills are worse.
- **Single asset, no portfolio construction**, no position sizing beyond
  -1/0/+1, no risk limits.
- **Survivorship bias** is untouched, and the bundled dataset has it by
  construction: those 505 companies are the ones still in the index in 2018.
- **One regime.** 2013-2018 contains no sustained bear market. Nothing here has
  been tested through a crash.
- **The multiple-comparisons problem is yours.** Nothing stops you from trying
  fifty symbols and reporting the best. The p-values above are not corrected for
  that, and the verdict text says so.

This is research tooling for studying market predictability. **It is not
investment advice and not a trading system.**

---

## finops

The other half of the repo: token accounting and cost control for Claude
applications — per-request cost, what prompt caching actually saved, context
budgeting, and an append-only usage ledger. See **[docs/finops.md](docs/finops.md)**.

```console
$ finops report --group-by tag:feature --since 7d
```
