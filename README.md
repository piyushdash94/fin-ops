# fin-ops

Two toolkits about the same discipline — measuring what something really costs,
and refusing to believe a number that hasn't been tested.

| Package | What it does |
|---|---|
| **[`marketrl`](#marketrl)** | Next-day stock return prediction with ML and RL, evaluated walk-forward and net of costs |
| **[`finops`](docs/finops.md)** | Token accounting and cost control for Claude-powered applications |

```bash
pip install -e ".[dev]"
python -m pytest        # 172 tests, no network required
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

### It is validated against a market with no edge

The generator can produce a market that is provably unpredictable
(`signal=0.0`) or one with a real planted edge (`signal=0.8`). Both are tested
on every run — `marketrl selftest` — because a backtester that cannot return a
null result will eventually tell you to trade on noise.

Here is the real output on an efficient market:

```console
$ marketrl demo --signal 0.0
FORECAST QUALITY  (predicting next-day log return)
  model                 dir.acc  p-value      IC      RMSE
  --------------------------------------------------------
  persistence             0.499   0.5419  -0.004   0.01147
  ridge                   0.493   0.7005  -0.018   0.00825
  random_forest           0.496   0.6238   0.008   0.00815
  gradient_boosting       0.501   0.4790   0.001   0.00834

STRATEGY PERFORMANCE  (net of costs, out-of-sample)
  strategy               return    CAGR  Sharpe   maxDD  turnover
  ---------------------------------------------------------------
  buy_and_hold            63.1%    8.9%    0.72   14.9%      0.2x
  reinforce              -11.6%   -2.1%   -0.10   19.0%     45.8x
  random_forest          -55.4%  -13.1%   -1.01   58.2%    133.9x
  ridge                  -62.9%  -15.8%   -1.26   65.6%    138.1x
  q_learning             -77.5%  -22.9%   -2.00   78.9%    191.7x

VERDICT
  No model's directional accuracy is distinguishable from a coin flip (p >= 0.05).
  Nothing beat buy-and-hold on risk-adjusted return after costs.
  That is the expected outcome on a liquid instrument. Do not
  re-run with new settings until something wins; that is how
  backtest overfitting happens.
```

Every model lands on ~50% and every strategy is destroyed by its own turnover.
Run the same command with `--signal 0.8` and all three models clear p < 0.01
with IC ≈ 0.13 and beat the benchmark — so the pipeline can find an edge; it
just correctly reports there isn't one above.

## Usage

```console
$ marketrl run --symbol AAPL --start 2010-01-01 --cost-bps 10
$ marketrl data --symbol MSFT --save msft.csv
$ marketrl selftest
```

```python
from marketrl import load, run_pipeline

print(run_pipeline(load("AAPL"), cost_bps=10.0).render())
```

Data comes from Yahoo Finance's chart API (used directly — one fewer dependency
than `yfinance`, and legible failures), with Stooq as a second free source and
local CSV for any other dataset. **A network failure raises**; it never
degrades to simulated prices. Synthetic data must be asked for by name, and
every report built on it carries a banner saying so.

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
- **Survivorship bias** is untouched — backtest a delisted ticker and you'll get
  nothing; backtest today's index members and you've already cheated.
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
