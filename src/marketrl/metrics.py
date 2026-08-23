"""Forecast and strategy metrics, with significance where it belongs.

Directional accuracy of 52% sounds like an edge and usually is not. Every
directional figure here ships with the probability of seeing it by chance, so a
result has to clear a bar before anyone acts on it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

__all__ = [
    "ForecastMetrics",
    "StrategyMetrics",
    "forecast_metrics",
    "strategy_metrics",
    "spearman_ic",
    "directional_accuracy",
    "binomial_p_value",
    "max_drawdown",
    "TRADING_DAYS",
]

TRADING_DAYS = 252


def _clean(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def directional_accuracy(y_true, y_pred) -> float:
    """Share of days where the predicted sign matched the realized sign.

    Days with a zero prediction are counted as misses: a model that abstains
    earns no credit for being right by not guessing.
    """
    y_true, y_pred = _clean(y_true, y_pred)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.sign(y_pred) == np.sign(y_true)))


def binomial_p_value(successes: int, trials: int, p: float = 0.5) -> float:
    """One-sided probability of at least this many successes by chance.

    Normal approximation with a continuity correction -- exact to well under
    the precision anyone should act on at the sample sizes involved here.
    """
    if trials <= 0:
        return float("nan")
    mean = trials * p
    sd = np.sqrt(trials * p * (1 - p))
    if sd == 0:
        return 1.0
    z = (successes - 0.5 - mean) / sd
    return float(1.0 - NormalDist().cdf(z))


def spearman_ic(y_true, y_pred) -> float:
    """Rank correlation between prediction and outcome (the information coefficient).

    Rank-based, so a few outlier days cannot manufacture a correlation. In
    equity forecasting a *genuine* daily IC of 0.03-0.05 is a real edge; values
    above 0.2 on daily data almost always mean a leak.
    """
    y_true, y_pred = _clean(y_true, y_pred)
    if y_true.size < 3:
        return float("nan")
    rank_true = pd.Series(y_true).rank().to_numpy()
    rank_pred = pd.Series(y_pred).rank().to_numpy()
    if rank_true.std() == 0 or rank_pred.std() == 0:
        return 0.0
    return float(np.corrcoef(rank_true, rank_pred)[0, 1])


def max_drawdown(equity) -> float:
    """Largest peak-to-trough fall of an equity curve, as a positive fraction."""
    equity = np.asarray(equity, dtype=float)
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / np.where(peak == 0, np.nan, peak)))


@dataclass
class ForecastMetrics:
    n: int
    rmse: float
    mae: float
    directional_accuracy: float
    directional_p_value: float
    information_coefficient: float
    r2: float

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.directional_p_value < alpha


def forecast_metrics(y_true, y_pred) -> ForecastMetrics:
    """Accuracy of a return forecast, including whether it beats a coin flip."""
    y_true, y_pred = _clean(y_true, y_pred)
    n = int(y_true.size)
    if n == 0:
        nan = float("nan")
        return ForecastMetrics(0, nan, nan, nan, nan, nan, nan)

    error = y_pred - y_true
    accuracy = directional_accuracy(y_true, y_pred)
    hits = int(round(accuracy * n))
    # R^2 against the sample mean. On daily returns this is normally slightly
    # negative even for a useful model; that is expected, not a bug.
    variance = float(np.sum((y_true - y_true.mean()) ** 2))
    return ForecastMetrics(
        n=n,
        rmse=float(np.sqrt(np.mean(error**2))),
        mae=float(np.mean(np.abs(error))),
        directional_accuracy=accuracy,
        directional_p_value=binomial_p_value(hits, n),
        information_coefficient=spearman_ic(y_true, y_pred),
        r2=float(1 - np.sum(error**2) / variance) if variance > 0 else float("nan"),
    )


@dataclass
class StrategyMetrics:
    n_days: int
    total_return: float
    cagr: float
    annual_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    exposure: float
    annual_turnover: float
    total_costs: float

    def as_dict(self) -> dict:
        return asdict(self)


def strategy_metrics(
    net_returns, positions=None, costs=None, *, periods_per_year: int = TRADING_DAYS
) -> StrategyMetrics:
    """Risk and return of a strategy from its per-period net returns."""
    returns = np.asarray(net_returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    n = int(returns.size)
    if n == 0:
        z = 0.0
        return StrategyMetrics(0, z, z, z, z, z, z, z, z, z, z, z)

    equity = np.cumprod(1.0 + returns)
    total_return = float(equity[-1] - 1.0)
    years = n / periods_per_year
    cagr = float(equity[-1] ** (1 / years) - 1) if years > 0 and equity[-1] > 0 else float("nan")

    volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if n > 1 else 0.0
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year)) if n > 1 and returns.std(ddof=1) > 0 else 0.0
    downside = returns[returns < 0]
    downside_dev = float(downside.std(ddof=1)) if downside.size > 1 else 0.0
    sortino = float(returns.mean() / downside_dev * np.sqrt(periods_per_year)) if downside_dev > 0 else 0.0

    drawdown = max_drawdown(equity)
    positions = np.asarray(positions, dtype=float) if positions is not None else np.ones(n)
    turnover = float(np.mean(np.abs(np.diff(positions, prepend=0.0))) * periods_per_year)

    return StrategyMetrics(
        n_days=n,
        total_return=total_return,
        cagr=cagr,
        annual_volatility=volatility,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=drawdown,
        calmar=float(cagr / drawdown) if drawdown > 0 and np.isfinite(cagr) else 0.0,
        hit_rate=float(np.mean(returns > 0)),
        exposure=float(np.mean(np.abs(positions))),
        annual_turnover=turnover,
        total_costs=float(np.sum(costs)) if costs is not None else 0.0,
    )
