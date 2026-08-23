"""The controls that decide whether any of this is trustworthy.

A backtesting framework has exactly two ways to be wrong: it can miss an edge
that exists, or invent one that doesn't. The second is far more expensive, so
the no-signal control is the load-bearing test in this file.
"""

import numpy as np
import pytest

from marketrl.data import synthetic_prices
from marketrl.pipeline import run_pipeline

pytestmark = pytest.mark.filterwarnings("ignore")

FAST = dict(train_size=500, test_size=250, agents=())


@pytest.fixture(scope="module")
def efficient_report():
    """An efficient market: nothing to find."""
    return run_pipeline(synthetic_prices(1500, seed=21, signal=0.0), **FAST)


@pytest.fixture(scope="module")
def inefficient_report():
    """A market with a real, planted edge."""
    return run_pipeline(synthetic_prices(1500, seed=21, signal=0.9), **FAST)


# ------------------------------------------------- the negative control
def test_no_spurious_edge_in_an_efficient_market(efficient_report):
    assert efficient_report.significant_forecasts() == []


def test_efficient_market_accuracy_is_near_a_coin_flip(efficient_report):
    for name, metrics in efficient_report.forecasts.items():
        if name == "zero":
            continue
        assert 0.44 < metrics.directional_accuracy < 0.56, name


def test_efficient_market_verdict_says_so(efficient_report):
    rendered = efficient_report.render()
    assert "distinguishable from a coin flip" in rendered
    assert "Nothing beat buy-and-hold" in rendered
    assert "overfitting" in rendered


def test_costs_sink_noise_strategies(efficient_report):
    """With no edge, trading must lose to buy-and-hold once costs are charged."""
    benchmark = efficient_report.strategies["buy_and_hold"]
    for name, result in efficient_report.strategies.items():
        if name == "buy_and_hold":
            continue
        assert result.metrics.sharpe < benchmark.metrics.sharpe, name


# ------------------------------------------------- the positive control
def test_a_real_edge_is_detected(inefficient_report):
    assert inefficient_report.significant_forecasts()


def test_a_real_edge_produces_a_tradeable_strategy(inefficient_report):
    assert inefficient_report.beats_benchmark()


def test_information_coefficient_rises_with_signal(efficient_report, inefficient_report):
    quiet = abs(efficient_report.forecasts["ridge"].information_coefficient)
    loud = abs(inefficient_report.forecasts["ridge"].information_coefficient)
    assert loud > quiet


# ------------------------------------------------------------ mechanics
def test_all_strategies_share_one_out_of_sample_window(inefficient_report):
    lengths = {len(r.net_returns) for r in inefficient_report.strategies.values()}
    assert len(lengths) == 1, "strategies were scored over different days"


def test_synthetic_provenance_is_impossible_to_miss(efficient_report):
    assert efficient_report.is_synthetic
    assert "SYNTHETIC PRICES" in efficient_report.render()


def test_report_serializes_to_json(inefficient_report):
    import json

    payload = json.loads(json.dumps(inefficient_report.as_dict(), default=str))
    assert payload["is_synthetic"] is True
    assert "buy_and_hold" in payload["strategies"]
    assert payload["n_test_days"] > 0


def test_higher_costs_reduce_net_performance():
    prices = synthetic_prices(1500, seed=21, signal=0.9)
    cheap = run_pipeline(prices, cost_bps=1.0, **FAST)
    dear = run_pipeline(prices, cost_bps=50.0, **FAST)
    assert (
        dear.strategies["ridge"].metrics.total_return
        < cheap.strategies["ridge"].metrics.total_return
    )


def test_long_only_never_takes_a_short():
    report = run_pipeline(
        synthetic_prices(1500, seed=21, signal=0.9), allow_short=False, **FAST
    )
    assert (report.strategies["ridge"].positions >= 0).all()


def test_threshold_reduces_turnover():
    prices = synthetic_prices(1500, seed=21, signal=0.9)
    always = run_pipeline(prices, threshold=0.0, **FAST)
    selective = run_pipeline(prices, threshold=0.01, **FAST)
    assert (
        selective.strategies["ridge"].metrics.annual_turnover
        < always.strategies["ridge"].metrics.annual_turnover
    )


def test_unknown_model_and_agent_are_rejected():
    prices = synthetic_prices(1500, seed=1)
    with pytest.raises(ValueError, match="unknown model"):
        run_pipeline(prices, models=("nope",), **FAST)
    with pytest.raises(ValueError, match="unknown agent"):
        run_pipeline(prices, models=("ridge",), train_size=500, test_size=250, agents=("nope",))


@pytest.mark.slow
def test_rl_agents_run_walk_forward_end_to_end():
    report = run_pipeline(
        synthetic_prices(1400, seed=5, signal=0.9),
        train_size=500, test_size=300, agents=("q_learning", "reinforce"),
    )
    assert {"q_learning", "reinforce"} <= set(report.strategies)
    for name in ("q_learning", "reinforce"):
        assert report.strategies[name].positions.isin([-1.0, 0.0, 1.0]).all()
