"""Walk-forward backtesting for both supervised and RL strategies.

Everything is evaluated the same way and over the same dates: train on the
past, act on the next block, never look back. Positions are taken at the close
of day ``t`` and earn day ``t+1``'s return, so no order is ever filled at a
price that was not yet known.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .agents import SupervisedPolicy
from .env import TradingEnv
from .metrics import StrategyMetrics, strategy_metrics
from .splits import walk_forward

__all__ = [
    "BacktestResult",
    "to_simple_returns",
    "backtest_positions",
    "walk_forward_rl",
    "positions_from_predictions",
]


def to_simple_returns(log_returns) -> np.ndarray:
    """Convert log returns to simple returns, which is what compounds."""
    return np.expm1(np.asarray(log_returns, dtype=float))


@dataclass
class BacktestResult:
    name: str
    positions: pd.Series
    net_returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    equity: pd.Series
    metrics: StrategyMetrics

    def as_dict(self) -> dict:
        return {"name": self.name, **self.metrics.as_dict()}

    def summary(self) -> str:
        m = self.metrics
        return (
            f"{self.name:<18} return {m.total_return:>8.1%}  CAGR {m.cagr:>7.1%}  "
            f"Sharpe {m.sharpe:>6.2f}  maxDD {m.max_drawdown:>6.1%}  "
            f"turnover {m.annual_turnover:>6.1f}x"
        )


def backtest_positions(
    positions: pd.Series,
    forward_returns: pd.Series,
    *,
    cost_bps: float = 10.0,
    name: str = "strategy",
    initial_position: float = 0.0,
) -> BacktestResult:
    """Score a position series against the returns it actually earned.

    ``positions[t]`` is the exposure decided at the close of day ``t``;
    ``forward_returns[t]`` is the simple return from ``t`` to ``t+1``. Costs are
    charged on every change in exposure, including the initial entry.
    """
    positions = positions.astype(float)
    forward_returns = forward_returns.reindex(positions.index).astype(float)

    turnover = positions.diff()
    turnover.iloc[0] = positions.iloc[0] - initial_position
    costs = turnover.abs() * (cost_bps / 10_000.0)

    gross = positions * forward_returns
    net = gross - costs
    equity = (1.0 + net.fillna(0.0)).cumprod()

    return BacktestResult(
        name=name,
        positions=positions,
        net_returns=net,
        gross_returns=gross,
        costs=costs,
        equity=equity,
        metrics=strategy_metrics(net.to_numpy(), positions.to_numpy(), costs.to_numpy()),
    )


def positions_from_predictions(
    predictions: pd.Series, *, threshold: float = 0.0, allow_short: bool = True
) -> pd.Series:
    """Map predicted returns to exposures, with a dead zone around zero.

    The threshold is what makes a supervised model cost-aware: below it the
    forecast is too weak to be worth paying the spread for.
    """
    positions = pd.Series(0.0, index=predictions.index)
    positions[predictions > threshold] = 1.0
    if allow_short:
        positions[predictions < -threshold] = -1.0
    return positions


def walk_forward_rl(
    features: pd.DataFrame,
    forward_returns: pd.Series,
    agent_factory: Callable[[], object],
    *,
    name: str = "rl",
    train_size: int = 750,
    test_size: int = 126,
    cost_bps: float = 10.0,
    allow_short: bool = True,
    expanding: bool = True,
    embargo: int = 1,
) -> BacktestResult:
    """Train an RL agent on each training window and act on the block that follows.

    A fresh agent is fitted per fold -- carrying one agent across folds would
    let it keep what it learned from data it is about to be tested on.
    """
    forward_returns = forward_returns.reindex(features.index).astype(float)
    values = features.to_numpy(dtype=float)
    returns = forward_returns.to_numpy(dtype=float)

    folds = list(
        walk_forward(
            len(features), train_size=train_size, test_size=test_size,
            expanding=expanding, embargo=embargo,
        )
    )
    if not folds:
        raise ValueError(
            f"{len(features)} rows is too few for walk-forward with "
            f"train_size={train_size}, test_size={test_size}"
        )

    chunks: list[pd.Series] = []
    for fold in folds:
        train_env = TradingEnv(
            values[fold.train], returns[fold.train],
            cost_bps=cost_bps, allow_short=allow_short,
        )
        agent = agent_factory()
        agent.fit(train_env)

        test_env = TradingEnv(
            values[fold.test], returns[fold.test],
            cost_bps=cost_bps, allow_short=allow_short,
        )
        if isinstance(agent, SupervisedPolicy):
            agent.reset()
        outcome = test_env.run(agent)
        chunks.append(pd.Series(outcome["positions"], index=features.index[fold.test]))

    positions = pd.concat(chunks)
    return backtest_positions(
        positions, forward_returns.loc[positions.index], cost_bps=cost_bps, name=name
    )
