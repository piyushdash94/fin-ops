import numpy as np
import pandas as pd
import pytest

from marketrl.backtest import (
    backtest_positions,
    positions_from_predictions,
    to_simple_returns,
    walk_forward_rl,
)
from marketrl.agents import QLearningAgent
from marketrl.metrics import (
    binomial_p_value,
    directional_accuracy,
    forecast_metrics,
    max_drawdown,
    spearman_ic,
    strategy_metrics,
)
from marketrl.splits import walk_forward


# ---------------------------------------------------------------- metrics
def test_directional_accuracy_edges():
    assert directional_accuracy([1, -1, 1], [2, -2, 3]) == 1.0
    assert directional_accuracy([1, -1], [-1, 1]) == 0.0
    # A zero prediction never matches a nonzero outcome: abstaining earns nothing.
    assert directional_accuracy([1, -1], [0, 0]) == 0.0


def test_binomial_p_value_behaviour():
    assert binomial_p_value(500, 1000) > 0.4          # a coin flip
    assert binomial_p_value(600, 1000) < 1e-6         # decisive
    assert binomial_p_value(55, 100) > 0.05           # same rate, too few trials
    assert np.isnan(binomial_p_value(0, 0))


def test_spearman_ic_is_rank_based():
    x = np.arange(50, dtype=float)
    assert spearman_ic(x, x**3) == pytest.approx(1.0)      # monotone -> perfect
    assert spearman_ic(x, -x) == pytest.approx(-1.0)
    assert spearman_ic(x, np.ones(50)) == 0.0              # constant -> none


def test_max_drawdown():
    assert max_drawdown([1.0, 2.0, 1.0]) == pytest.approx(0.5)
    assert max_drawdown([1.0, 1.1, 1.2]) == pytest.approx(0.0)


def test_forecast_metrics_on_a_perfect_forecast():
    y = np.array([0.01, -0.02, 0.03, -0.01])
    m = forecast_metrics(y, y)
    assert m.rmse == 0.0
    assert m.directional_accuracy == 1.0
    assert m.r2 == pytest.approx(1.0)


def test_forecast_metrics_handles_empty_input():
    assert forecast_metrics([], []).n == 0


def test_sharpe_matches_the_definition():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0005, 0.01, 5000)
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert strategy_metrics(returns).sharpe == pytest.approx(expected)


def test_turnover_counts_position_changes():
    # Flip between long and short every day: 2 units of turnover per day.
    positions = np.tile([1.0, -1.0], 126)
    metrics = strategy_metrics(np.zeros(252), positions)
    assert metrics.annual_turnover == pytest.approx(2 * 252, rel=0.02)

    held = np.ones(252)
    assert strategy_metrics(np.zeros(252), held).annual_turnover == pytest.approx(1.0)


def test_strategy_metrics_on_empty_input():
    assert strategy_metrics([]).n_days == 0


# --------------------------------------------------------------- backtest
def test_to_simple_returns_inverts_logs():
    log_returns = np.array([0.01, -0.02, 0.0])
    np.testing.assert_allclose(
        np.log1p(to_simple_returns(log_returns)), log_returns, atol=1e-12
    )


def test_backtest_charges_the_initial_entry():
    index = pd.bdate_range("2024-01-01", periods=3)
    result = backtest_positions(
        pd.Series(1.0, index=index), pd.Series(0.0, index=index), cost_bps=10
    )
    assert result.costs.iloc[0] == pytest.approx(0.001)
    assert result.costs.iloc[1:].sum() == 0.0


def test_backtest_equity_compounds():
    index = pd.bdate_range("2024-01-01", periods=3)
    result = backtest_positions(
        pd.Series(1.0, index=index),
        pd.Series([0.10, 0.10, 0.10], index=index),
        cost_bps=0,
    )
    assert result.equity.iloc[-1] == pytest.approx(1.1**3)
    assert result.metrics.total_return == pytest.approx(1.1**3 - 1)


def test_short_position_earns_the_inverse():
    index = pd.bdate_range("2024-01-01", periods=2)
    result = backtest_positions(
        pd.Series(-1.0, index=index), pd.Series(0.05, index=index), cost_bps=0
    )
    assert result.gross_returns.iloc[0] == pytest.approx(-0.05)


def test_costs_scale_with_position_change():
    index = pd.bdate_range("2024-01-01", periods=2)
    # Long to short is 2 units of turnover, so twice the entry cost.
    result = backtest_positions(
        pd.Series([1.0, -1.0], index=index), pd.Series(0.0, index=index), cost_bps=10
    )
    assert result.costs.iloc[1] == pytest.approx(0.002)


def test_positions_from_predictions_threshold():
    predictions = pd.Series([0.01, -0.01, 0.0001, -0.0001])
    positions = positions_from_predictions(predictions, threshold=0.005)
    assert positions.tolist() == [1.0, -1.0, 0.0, 0.0]


def test_positions_long_only_never_shorts():
    predictions = pd.Series([-0.05, 0.05])
    positions = positions_from_predictions(predictions, allow_short=False)
    assert positions.tolist() == [0.0, 1.0]


def test_walk_forward_rl_predicts_only_out_of_sample():
    rng = np.random.default_rng(0)
    n = 900
    features = pd.DataFrame(
        rng.standard_normal((n, 3)), index=pd.bdate_range("2015-01-01", periods=n)
    )
    forward = pd.Series(rng.standard_normal(n) * 0.01, index=features.index)

    result = walk_forward_rl(
        features, forward, lambda: QLearningAgent(episodes=5),
        train_size=400, test_size=200,
    )
    folds = list(walk_forward(n, train_size=400, test_size=200))
    expected = sum(len(f.test) for f in folds)
    assert len(result.positions) == expected
    # The first test day starts after the first training window plus embargo.
    assert result.positions.index[0] == features.index[folds[0].test[0]]


def test_walk_forward_rl_rejects_too_short_a_series():
    features = pd.DataFrame(np.zeros((50, 2)), index=pd.bdate_range("2020-01-01", periods=50))
    forward = pd.Series(np.zeros(50), index=features.index)
    with pytest.raises(ValueError, match="too few"):
        walk_forward_rl(features, forward, lambda: QLearningAgent(), train_size=400, test_size=100)
