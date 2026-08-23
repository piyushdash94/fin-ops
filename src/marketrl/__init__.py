"""marketrl -- next-day return prediction with ML and RL, evaluated honestly.

The pipeline is ordinary: technical features, regularized supervised models,
two reinforcement-learning agents, a walk-forward backtest. What it is built
around is the part usually left out -- transaction costs inside the reward,
a mechanical no-lookahead check, significance tests on every directional
claim, and a report that says plainly when nothing worked.

    from marketrl import load, run_pipeline
    print(run_pipeline(load("AAPL")).render())

Research tooling. Not investment advice.
"""

from .agents import (
    AlwaysFlatPolicy,
    BuyAndHoldPolicy,
    QLearningAgent,
    RandomPolicy,
    ReinforceAgent,
    SupervisedPolicy,
)
from .backtest import (
    BacktestResult,
    backtest_positions,
    positions_from_predictions,
    to_simple_returns,
    walk_forward_rl,
)
from .data import (
    DataError, load, load_csv, load_sp500, load_stooq, load_yahoo,
    sp500_symbols, synthetic_prices,
)
from .env import TradingEnv
from .features import FEATURE_COLUMNS, assert_no_lookahead, make_dataset, make_features
from .metrics import ForecastMetrics, StrategyMetrics, forecast_metrics, strategy_metrics
from .pipeline import PipelineReport, run_pipeline
from .scan import ScanReport, scan_symbols
from .splits import walk_forward
from .supervised import MODELS, compare_models, walk_forward_predict

__version__ = "0.1.0"

__all__ = [
    "AlwaysFlatPolicy", "BacktestResult", "BuyAndHoldPolicy", "DataError",
    "FEATURE_COLUMNS", "ForecastMetrics", "MODELS", "PipelineReport",
    "QLearningAgent", "RandomPolicy", "ReinforceAgent", "StrategyMetrics",
    "SupervisedPolicy", "TradingEnv", "assert_no_lookahead", "backtest_positions",
    "ScanReport", "compare_models", "forecast_metrics", "load", "load_csv",
    "load_sp500", "load_stooq", "sp500_symbols", "scan_symbols",
    "load_yahoo", "make_dataset", "make_features", "positions_from_predictions",
    "run_pipeline", "strategy_metrics", "synthetic_prices", "to_simple_returns",
    "walk_forward", "walk_forward_predict", "walk_forward_rl", "__version__",
]
