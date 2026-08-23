"""End-to-end: load prices, forecast, trade, and report what actually happened.

The report is built to make a null result legible. Most of the time, on most
liquid instruments, nothing here will beat buy-and-hold after costs -- that is
what market efficiency means, and a framework that cannot say so plainly is
worse than useless because it will eventually tell you to trade.

Everything below is out-of-sample, walk-forward, and net of transaction costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .agents import QLearningAgent, ReinforceAgent
from .backtest import (
    BacktestResult,
    backtest_positions,
    positions_from_predictions,
    to_simple_returns,
    walk_forward_rl,
)
from .features import make_dataset
from .metrics import ForecastMetrics
from .supervised import BASELINE_MODELS, MODELS, walk_forward_predict

__all__ = ["PipelineReport", "run_pipeline", "DEFAULT_MODELS", "DEFAULT_AGENTS"]

DEFAULT_MODELS = ("zero", "persistence", "ridge", "random_forest", "gradient_boosting")
DEFAULT_AGENTS = ("q_learning", "reinforce")

_AGENT_FACTORIES = {
    "q_learning": lambda seed: (lambda: QLearningAgent(episodes=60, seed=seed)),
    "reinforce": lambda seed: (lambda: ReinforceAgent(episodes=150, seed=seed)),
}


@dataclass
class PipelineReport:
    symbol: str
    source: str
    is_synthetic: bool
    start: pd.Timestamp
    end: pd.Timestamp
    n_test_days: int
    cost_bps: float
    forecasts: dict[str, ForecastMetrics] = field(default_factory=dict)
    strategies: dict[str, BacktestResult] = field(default_factory=dict)

    @property
    def benchmark(self) -> BacktestResult | None:
        return self.strategies.get("buy_and_hold")

    def beats_benchmark(self) -> list[str]:
        """Strategies with a higher Sharpe than buy-and-hold, net of costs."""
        benchmark = self.benchmark
        if benchmark is None:
            return []
        return [
            name
            for name, result in self.strategies.items()
            if name != "buy_and_hold" and result.metrics.sharpe > benchmark.metrics.sharpe
        ]

    def significant_forecasts(self, alpha: float = 0.05) -> list[str]:
        return [
            name
            for name, metrics in self.forecasts.items()
            if name not in BASELINE_MODELS and metrics.directional_p_value < alpha
        ]

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "source": self.source,
            "is_synthetic": self.is_synthetic,
            "start": str(self.start.date()),
            "end": str(self.end.date()),
            "n_test_days": self.n_test_days,
            "cost_bps": self.cost_bps,
            "forecasts": {k: v.as_dict() for k, v in self.forecasts.items()},
            "strategies": {k: v.as_dict() for k, v in self.strategies.items()},
            "beats_benchmark": self.beats_benchmark(),
            "significant_forecasts": self.significant_forecasts(),
        }

    def render(self) -> str:
        lines: list[str] = []
        if self.is_synthetic:
            lines += [
                "!" * 72,
                "!! SYNTHETIC PRICES -- these results describe a simulation, not a",
                "!! market. Nothing here is evidence about any real instrument.",
                "!" * 72,
                "",
            ]

        lines += [
            f"{self.symbol}  ({self.source})",
            f"out-of-sample {self.start.date()} to {self.end.date()}  "
            f"({self.n_test_days:,} trading days, {self.cost_bps:g} bps per unit traded)",
            "",
            "FORECAST QUALITY  (predicting next-day log return)",
            f"  {'model':<20}{'dir.acc':>9}{'p-value':>9}{'IC':>8}{'RMSE':>10}",
            "  " + "-" * 56,
        ]
        for name, metrics in self.forecasts.items():
            flag = " *" if metrics.directional_p_value < 0.05 else ""
            lines.append(
                f"  {name:<20}{metrics.directional_accuracy:>9.3f}"
                f"{metrics.directional_p_value:>9.4f}{metrics.information_coefficient:>8.3f}"
                f"{metrics.rmse:>10.5f}{flag}"
            )

        lines += [
            "",
            "STRATEGY PERFORMANCE  (net of costs, out-of-sample)",
            f"  {'strategy':<20}{'return':>9}{'CAGR':>8}{'Sharpe':>8}"
            f"{'maxDD':>8}{'turnover':>10}",
            "  " + "-" * 63,
        ]
        for name, result in sorted(
            self.strategies.items(), key=lambda kv: -kv[1].metrics.sharpe
        ):
            m = result.metrics
            lines.append(
                f"  {name:<20}{m.total_return:>9.1%}{m.cagr:>8.1%}{m.sharpe:>8.2f}"
                f"{m.max_drawdown:>8.1%}{m.annual_turnover:>9.1f}x"
            )

        lines += ["", "VERDICT"]
        winners = self.beats_benchmark()
        significant = self.significant_forecasts()

        if not significant:
            lines.append(
                "  No model's directional accuracy is distinguishable from a coin "
                "flip (p >= 0.05)."
            )
        else:
            lines.append(
                f"  Statistically significant direction calls: {', '.join(significant)}."
            )
            lines.append(
                "  Significant is not the same as tradeable -- check the turnover "
                "column before believing it."
            )

        if not winners:
            lines.append("  Nothing beat buy-and-hold on risk-adjusted return after costs.")
            lines.append("  That is the expected outcome on a liquid instrument. Do not")
            lines.append("  re-run with new settings until something wins; that is how")
            lines.append("  backtest overfitting happens.")
        else:
            lines.append(f"  Beat buy-and-hold on Sharpe: {', '.join(winners)}.")
            lines.append(
                "  Before acting: confirm it survives other symbols, other periods, "
                "and a higher cost assumption."
            )
        return "\n".join(lines)


def run_pipeline(
    prices: pd.DataFrame,
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    agents: tuple[str, ...] = DEFAULT_AGENTS,
    train_size: int = 750,
    test_size: int = 126,
    cost_bps: float = 10.0,
    threshold: float = 0.0,
    allow_short: bool = True,
    expanding: bool = True,
    seed: int = 0,
) -> PipelineReport:
    """Run the full comparison on one price series.

    Supervised models, RL agents and the buy-and-hold benchmark are all scored
    over the identical out-of-sample window, so the comparison is like-for-like.
    """
    features, target, _ = make_dataset(prices)
    forward_returns = pd.Series(to_simple_returns(target), index=target.index)

    forecasts: dict[str, ForecastMetrics] = {}
    strategies: dict[str, BacktestResult] = {}
    common_index: pd.Index | None = None

    for name in models:
        if name not in MODELS:
            raise ValueError(f"unknown model {name!r}; available: {', '.join(MODELS)}")
        result = walk_forward_predict(
            features, target, name,
            train_size=train_size, test_size=test_size, expanding=expanding,
        )
        forecasts[name] = result.metrics
        common_index = result.predictions.index if common_index is None else common_index

        if name in BASELINE_MODELS:
            continue  # a constant forecast is not a strategy worth backtesting
        positions = positions_from_predictions(
            result.predictions, threshold=threshold, allow_short=allow_short
        )
        strategies[name] = backtest_positions(
            positions, forward_returns.loc[positions.index], cost_bps=cost_bps, name=name
        )

    if common_index is None:
        raise ValueError("no models were run")

    for name in agents:
        if name not in _AGENT_FACTORIES:
            raise ValueError(
                f"unknown agent {name!r}; available: {', '.join(_AGENT_FACTORIES)}"
            )
        strategies[name] = walk_forward_rl(
            features, forward_returns, _AGENT_FACTORIES[name](seed),
            name=name, train_size=train_size, test_size=test_size,
            cost_bps=cost_bps, allow_short=allow_short, expanding=expanding,
        )

    # Benchmarks over exactly the same days as everything else.
    strategies["buy_and_hold"] = backtest_positions(
        pd.Series(1.0, index=common_index),
        forward_returns.loc[common_index],
        cost_bps=cost_bps, name="buy_and_hold",
    )

    return PipelineReport(
        symbol=str(prices.attrs.get("symbol", "?")),
        source=str(prices.attrs.get("source", "?")),
        is_synthetic=bool(prices.attrs.get("is_synthetic", False)),
        start=common_index[0],
        end=common_index[-1],
        n_test_days=len(common_index),
        cost_bps=cost_bps,
        forecasts=forecasts,
        strategies=strategies,
    )
