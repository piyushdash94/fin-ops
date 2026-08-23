"""Scan many symbols at once, and correct for having done so.

This module exists because of the single most common way quantitative research
fools itself: test one strategy on fifty stocks, find the three where p < 0.05,
and report those three. At a 5% threshold, fifty independent coin flips produce
2.5 "significant" results on average. Finding some is not evidence; finding
*more than chance predicts* is.

So the scan reports the whole distribution, states how many discoveries chance
alone would have produced, and applies Benjamini-Hochberg and Bonferroni
corrections before letting anything be called a finding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import data as _data
from .backtest import backtest_positions, positions_from_predictions, to_simple_returns
from .data import DataError
from .features import make_dataset
from .supervised import walk_forward_predict

__all__ = ["SymbolResult", "ScanReport", "scan_symbols", "benjamini_hochberg"]


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction; returns a boolean 'discovery' mask.

    Controls the expected *proportion* of false positives among rejections,
    which is the right criterion when scanning hundreds of symbols -- Bonferroni
    controls the probability of even one, and at n=500 is so strict that a real
    edge would have to be enormous to survive it.
    """
    p_values = np.asarray(p_values, dtype=float)
    n = p_values.size
    if n == 0:
        return np.zeros(0, dtype=bool)

    order = np.argsort(p_values)
    ranked = p_values[order]
    thresholds = alpha * np.arange(1, n + 1) / n
    passing = np.nonzero(ranked <= thresholds)[0]

    discoveries = np.zeros(n, dtype=bool)
    if passing.size:
        discoveries[order[: passing[-1] + 1]] = True
    return discoveries


@dataclass
class SymbolResult:
    symbol: str
    n_test_days: int
    directional_accuracy: float
    p_value: float
    information_coefficient: float
    strategy_sharpe: float
    benchmark_sharpe: float
    strategy_return: float
    benchmark_return: float
    annual_turnover: float

    @property
    def excess_sharpe(self) -> float:
        return self.strategy_sharpe - self.benchmark_sharpe

    def as_dict(self) -> dict:
        return {**asdict(self), "excess_sharpe": self.excess_sharpe}


@dataclass
class ScanReport:
    model: str
    source: str
    cost_bps: float
    alpha: float
    results: list[SymbolResult] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    min_test_days: int = 0

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def p_values(self) -> np.ndarray:
        return np.array([r.p_value for r in self.results])

    @property
    def n_nominally_significant(self) -> int:
        return int(np.sum(self.p_values < self.alpha))

    @property
    def n_expected_by_chance(self) -> float:
        return self.alpha * self.n

    @property
    def chance_sd(self) -> float:
        """Spread in directional accuracy that coin flips alone would produce.

        For n independent days at p=0.5 the sample proportion has standard
        deviation sqrt(0.25/n). If the observed spread across symbols matches
        this, the variation between "good" and "bad" symbols is entirely noise
        -- there is no per-symbol skill to rank.
        """
        if not self.results:
            return 0.0
        days = np.array([r.n_test_days for r in self.results], dtype=float)
        return float(np.sqrt(0.25 / np.median(days)))

    def discoveries(self) -> list[SymbolResult]:
        """Symbols surviving Benjamini-Hochberg FDR control."""
        if not self.results:
            return []
        mask = benjamini_hochberg(self.p_values, self.alpha)
        return [r for r, keep in zip(self.results, mask) if keep]

    def bonferroni_survivors(self) -> list[SymbolResult]:
        if not self.results:
            return []
        return [r for r in self.results if r.p_value < self.alpha / self.n]

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "source": self.source,
            "cost_bps": self.cost_bps,
            "n_symbols": self.n,
            "min_test_days": self.min_test_days,
            "mean_directional_accuracy": float(
                np.mean([r.directional_accuracy for r in self.results])
            ) if self.results else float("nan"),
            "n_nominally_significant": self.n_nominally_significant,
            "n_expected_by_chance": self.n_expected_by_chance,
            "observed_sd": float(np.std([r.directional_accuracy for r in self.results]))
            if self.results else float("nan"),
            "chance_sd": self.chance_sd,
            "fdr_discoveries": [r.symbol for r in self.discoveries()],
            "bonferroni_survivors": [r.symbol for r in self.bonferroni_survivors()],
            "n_beating_benchmark": sum(1 for r in self.results if r.excess_sharpe > 0),
            "results": [r.as_dict() for r in self.results],
            "failures": self.failures,
        }

    def render(self, *, top: int = 10) -> str:
        if not self.results:
            return "no symbols completed; " + "; ".join(
                f"{k}: {v}" for k, v in list(self.failures.items())[:5]
            )

        accuracy = np.array([r.directional_accuracy for r in self.results])
        excess = np.array([r.excess_sharpe for r in self.results])
        beat = int(np.sum(excess > 0))

        lines = [
            f"SCAN  model={self.model}  source={self.source}  "
            f"{self.n} symbols  {self.cost_bps:g} bps",
            f"      {int(np.median([r.n_test_days for r in self.results]))} "
            f"out-of-sample days per symbol (median)",
            "",
            "DIRECTIONAL ACCURACY ACROSS SYMBOLS",
            f"  mean {accuracy.mean():.4f}   median {np.median(accuracy):.4f}   "
            f"sd {accuracy.std():.4f}",
            f"  range {accuracy.min():.3f} to {accuracy.max():.3f}",
            f"  sd predicted by chance alone {self.chance_sd:.4f}   "
            f"(observed/predicted = {accuracy.std() / self.chance_sd:.2f}x)"
            if self.chance_sd > 0 else "  (sample too small to estimate a chance spread)",
            "",
            "MULTIPLE COMPARISONS",
            f"  nominally significant (p < {self.alpha})   "
            f"{self.n_nominally_significant} of {self.n}",
            f"  expected from chance alone                {self.n_expected_by_chance:.1f}",
        ]

        discoveries = self.discoveries()
        bonferroni = self.bonferroni_survivors()
        lines += [
            f"  survive Benjamini-Hochberg FDR            {len(discoveries)}"
            + (f"  ({', '.join(r.symbol for r in discoveries[:8])})" if discoveries else ""),
            f"  survive Bonferroni                        {len(bonferroni)}"
            + (f"  ({', '.join(r.symbol for r in bonferroni[:8])})" if bonferroni else ""),
            "",
            "VERSUS BUY-AND-HOLD (net of costs)",
            f"  beat benchmark on Sharpe                  {beat} of {self.n} "
            f"({beat / self.n:.0%})",
            f"  mean excess Sharpe                        {excess.mean():+.3f}",
            f"  median annual turnover                    "
            f"{np.median([r.annual_turnover for r in self.results]):.0f}x",
            "",
            f"BEST {min(top, self.n)} BY p-VALUE",
            f"  {'symbol':<8}{'dir.acc':>9}{'p':>9}{'IC':>8}"
            f"{'Sharpe':>9}{'bench':>8}{'excess':>9}",
            "  " + "-" * 60,
        ]
        for r in sorted(self.results, key=lambda r: r.p_value)[:top]:
            lines.append(
                f"  {r.symbol:<8}{r.directional_accuracy:>9.3f}{r.p_value:>9.4f}"
                f"{r.information_coefficient:>8.3f}{r.strategy_sharpe:>9.2f}"
                f"{r.benchmark_sharpe:>8.2f}{r.excess_sharpe:>+9.2f}"
            )

        lines += ["", "VERDICT"]
        if not discoveries:
            lines += [
                f"  {self.n_nominally_significant} symbols cleared p < {self.alpha} "
                f"individually, but chance alone predicts "
                f"{self.n_expected_by_chance:.1f}.",
                "  None survive correction for having tested "
                f"{self.n} of them. There is no",
                "  evidence of next-day predictability here.",
            ]
        else:
            lines += [
                f"  {len(discoveries)} symbol(s) survive FDR correction: "
                f"{', '.join(r.symbol for r in discoveries[:8])}.",
                "  Before believing it: re-test on a different period, and check",
                "  whether the excess Sharpe survives a higher cost assumption.",
            ]
        if self.failures:
            lines.append(f"  ({len(self.failures)} symbols failed to load or were too short)")
        return "\n".join(lines)


def scan_symbols(
    symbols: list[str],
    *,
    source: str = "sp500",
    model: str = "ridge",
    start=None,
    end=None,
    train_size: int = 500,
    test_size: int = 126,
    cost_bps: float = 10.0,
    threshold: float = 0.0,
    allow_short: bool = True,
    alpha: float = 0.05,
    min_test_days: int = 250,
    progress: bool = False,
) -> ScanReport:
    """Run one model walk-forward across many symbols and pool the evidence.

    ``start`` defaults to None (use every available bar) rather than inheriting
    :func:`marketrl.data.load`'s default start date -- silently truncating the
    history would shrink each symbol's test set, and small test sets are how a
    scan manufactures 70% accuracy out of noise.

    ``min_test_days`` drops symbols whose out-of-sample window is too short to
    say anything. At 28 test days a coin flip clears p < 0.05 about one time in
    twenty and posts a Sharpe above 3; including such symbols corrupts both the
    distribution and the "best by p-value" table.
    """
    report = ScanReport(model=model, source=source, cost_bps=cost_bps, alpha=alpha)
    report.min_test_days = min_test_days

    for i, symbol in enumerate(symbols, 1):
        try:
            prices = _data.load(symbol, start, end, source=source)
            features, target, _ = make_dataset(prices)
            forward = pd.Series(to_simple_returns(target), index=target.index)

            outcome = walk_forward_predict(
                features, target, model, train_size=train_size, test_size=test_size
            )
            positions = positions_from_predictions(
                outcome.predictions, threshold=threshold, allow_short=allow_short
            )
            index = positions.index
            if len(index) < min_test_days:
                report.failures[symbol] = (
                    f"only {len(index)} out-of-sample days (need {min_test_days})"
                )
                continue
            strategy = backtest_positions(
                positions, forward.loc[index], cost_bps=cost_bps, name=model
            )
            benchmark = backtest_positions(
                pd.Series(1.0, index=index), forward.loc[index],
                cost_bps=cost_bps, name="buy_and_hold",
            )
        except (DataError, ValueError) as err:
            report.failures[symbol] = str(err)[:120]
            continue

        report.results.append(
            SymbolResult(
                symbol=symbol,
                n_test_days=len(index),
                directional_accuracy=outcome.metrics.directional_accuracy,
                p_value=outcome.metrics.directional_p_value,
                information_coefficient=outcome.metrics.information_coefficient,
                strategy_sharpe=strategy.metrics.sharpe,
                benchmark_sharpe=benchmark.metrics.sharpe,
                strategy_return=strategy.metrics.total_return,
                benchmark_return=benchmark.metrics.total_return,
                annual_turnover=strategy.metrics.annual_turnover,
            )
        )
        if progress and i % 25 == 0:
            print(f"  ... {i}/{len(symbols)} symbols", flush=True)

    return report
