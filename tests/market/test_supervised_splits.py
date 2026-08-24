import numpy as np
import pandas as pd
import pytest

from marketrl.data import synthetic_prices
from marketrl.features import make_dataset
from marketrl.splits import n_folds, walk_forward
from marketrl.supervised import (
    MODELS,
    PersistenceForecast,
    ZeroForecast,
    build_model,
    compare_models,
    walk_forward_predict,
)


# ------------------------------------------------------------------ splits
def test_train_always_precedes_test_with_an_embargo():
    for fold in walk_forward(2000, train_size=500, test_size=250, embargo=1):
        assert fold.train[-1] < fold.test[0]
        assert fold.test[0] - fold.train[-1] > 1  # embargoed, not merely adjacent


def test_expanding_windows_grow_and_rolling_windows_do_not():
    expanding = list(walk_forward(2000, train_size=500, test_size=250))
    assert len(expanding[1].train) > len(expanding[0].train)
    assert all(f.train[0] == 0 for f in expanding)

    rolling = list(walk_forward(2000, train_size=500, test_size=250, expanding=False))
    assert all(len(f.train) == 500 for f in rolling)
    assert rolling[1].train[0] > rolling[0].train[0]


def test_folds_never_exceed_the_series():
    for fold in walk_forward(1000, train_size=500, test_size=250):
        assert fold.test[-1] < 1000


def test_test_blocks_do_not_overlap():
    folds = list(walk_forward(3000, train_size=500, test_size=250))
    seen = np.concatenate([f.test for f in folds])
    assert len(seen) == len(set(seen.tolist()))


def test_no_folds_when_the_series_is_too_short():
    assert n_folds(100, train_size=500, test_size=250) == 0


def test_invalid_split_parameters():
    with pytest.raises(ValueError):
        list(walk_forward(1000, train_size=0))
    with pytest.raises(ValueError):
        list(walk_forward(1000, embargo=-1))


# ------------------------------------------------------------- supervised
@pytest.fixture(scope="module")
def dataset():
    X, y, _ = make_dataset(synthetic_prices(1400, seed=17, signal=0.8))
    return X, y


def test_zero_forecast_predicts_zero():
    assert np.all(ZeroForecast().fit(None).predict(np.zeros((5, 2))) == 0.0)


def test_persistence_reads_the_last_return():
    X = pd.DataFrame({"ret_lag_1": [0.01, -0.02], "other": [9.0, 9.0]})
    model = PersistenceForecast().fit(X)
    np.testing.assert_allclose(model.predict(X), [0.01, -0.02])


def test_every_registered_model_builds_and_fits(dataset):
    X, y = dataset
    for name in MODELS:
        model = build_model(name)
        model.fit(X.iloc[:300], y.iloc[:300])
        predictions = model.predict(X.iloc[300:350])
        assert len(predictions) == 50
        assert np.isfinite(predictions).all(), name


def test_unknown_model_name():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("magic")


def test_predictions_are_out_of_sample_only(dataset):
    X, y = dataset
    result = walk_forward_predict(X, y, "ridge", train_size=500, test_size=250)
    folds = list(walk_forward(len(X), train_size=500, test_size=250))
    assert len(result.predictions) == sum(len(f.test) for f in folds)
    assert result.n_folds == len(folds)
    # Nothing before the first test block is ever predicted.
    assert result.predictions.index[0] == X.index[folds[0].test[0]]


def test_predictions_align_with_actuals(dataset):
    X, y = dataset
    result = walk_forward_predict(X, y, "ridge", train_size=500, test_size=250)
    assert result.predictions.index.equals(result.actuals.index)
    pd.testing.assert_series_equal(result.actuals, y.loc[result.predictions.index])


def test_models_recover_a_planted_edge(dataset):
    X, y = dataset
    results = compare_models(X, y, ["zero", "ridge"], train_size=500, test_size=250)
    assert results["ridge"].metrics.directional_p_value < 0.05
    assert results["ridge"].metrics.information_coefficient > 0.05


def test_models_find_nothing_when_there_is_nothing():
    X, y = make_dataset(synthetic_prices(1400, seed=17, signal=0.0))[:2]
    result = walk_forward_predict(X, y, "ridge", train_size=500, test_size=250)
    assert result.metrics.directional_p_value > 0.05


def test_walk_forward_rejects_too_short_a_series(dataset):
    X, y = dataset
    with pytest.raises(ValueError, match="too few"):
        walk_forward_predict(X, y, "ridge", train_size=10_000, test_size=250)


def test_a_custom_estimator_factory_is_accepted(dataset):
    X, y = dataset
    result = walk_forward_predict(X, y, lambda: ZeroForecast(), train_size=500, test_size=250)
    assert (result.predictions == 0).all()
