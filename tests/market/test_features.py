import numpy as np
import pandas as pd
import pytest

from marketrl.data import synthetic_prices
from marketrl.features import (
    TARGET,
    assert_no_lookahead,
    make_dataset,
    make_features,
)


@pytest.fixture(scope="module")
def prices():
    return synthetic_prices(1200, seed=13)


def test_features_contain_no_lookahead(prices):
    # The property that matters most in the whole package.
    assert_no_lookahead(prices)


def test_lookahead_check_catches_a_deliberate_leak(prices):
    import marketrl.features as features_module

    original = features_module.make_features

    def leaky(frame, price_column="adj_close"):
        out = original(frame, price_column=price_column)
        out["cheat"] = frame[price_column].shift(-1)  # tomorrow's price
        return out

    features_module.make_features = leaky
    try:
        with pytest.raises(AssertionError, match="lookahead detected"):
            assert_no_lookahead(prices)
    finally:
        features_module.make_features = original


def test_target_is_exactly_the_next_day_log_return(prices):
    X, y, _ = make_dataset(prices)
    close = prices["adj_close"]
    expected = np.log(close.shift(-1) / close).loc[y.index]
    np.testing.assert_allclose(y.to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_dataset_is_aligned_and_finite(prices):
    X, y, forward = make_dataset(prices)
    assert len(X) == len(y) == len(forward)
    assert X.index.equals(y.index)
    assert np.isfinite(X.to_numpy()).all()
    assert np.isfinite(y.to_numpy()).all()
    assert TARGET not in X.columns


def test_multi_day_horizon(prices):
    _, y, _ = make_dataset(prices, horizon=5)
    close = prices["adj_close"]
    expected = np.log(close.shift(-5) / close).loc[y.index]
    np.testing.assert_allclose(y.to_numpy(), expected.to_numpy(), rtol=1e-12)


def test_invalid_horizon_rejected(prices):
    with pytest.raises(ValueError):
        make_dataset(prices, horizon=0)


def test_too_short_a_series_raises_a_useful_error():
    with pytest.raises(ValueError, match="too few"):
        make_dataset(synthetic_prices(30, seed=1))


def test_features_are_scale_invariant(prices):
    """Doubling every price must not move a scale-free feature."""
    doubled = prices.copy()
    for column in ("open", "high", "low", "close", "adj_close"):
        doubled[column] *= 2.0

    base = make_features(prices).dropna()
    scaled = make_features(doubled).dropna()
    for column in ("ret_lag_1", "rsi_14", "ma_gap_21", "momentum_21", "intraday_range"):
        np.testing.assert_allclose(
            base[column].to_numpy(), scaled[column].to_numpy(), rtol=1e-8, atol=1e-10
        )


def test_rsi_stays_in_range(prices):
    rsi = make_features(prices)["rsi_14"].dropna()
    assert rsi.between(0.0, 1.0).all()
