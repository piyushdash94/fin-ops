"""Supervised next-day return models, evaluated walk-forward.

The model zoo is deliberately biased toward heavy regularization. With ~25
features and a signal-to-noise ratio near zero, an unconstrained learner will
fit the noise perfectly and generalize worse than predicting zero -- so the
default ridge penalty is large, trees are shallow, and every model is compared
against baselines that cost nothing to run.

A baseline that beats your model is not an embarrassment; it is the correct
result, and this module is built to surface it rather than hide it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import ForecastMetrics, forecast_metrics
from .splits import Fold, walk_forward

__all__ = [
    "MODELS",
    "ZeroForecast",
    "MeanForecast",
    "PersistenceForecast",
    "build_model",
    "WalkForwardResult",
    "walk_forward_predict",
    "compare_models",
]


class ZeroForecast(BaseEstimator, RegressorMixin):
    """Predict zero return. The benchmark every model must clear.

    On daily equity data this is a genuinely strong forecast: it has the lowest
    achievable RMSE among constant predictors that don't bet on drift.
    """

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class MeanForecast(BaseEstimator, RegressorMixin):
    """Predict the training-window mean return (i.e. bet on drift only)."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), getattr(self, "mean_", 0.0))


class PersistenceForecast(BaseEstimator, RegressorMixin):
    """Predict that tomorrow repeats today (naive momentum).

    Reads the most recent return straight out of the ``ret_lag_1`` feature, so
    it needs no fitting and represents the "no model at all" trend-follower.
    """

    def __init__(self, feature: str = "ret_lag_1"):
        self.feature = feature

    def fit(self, X, y=None):
        self.column_ = list(X.columns).index(self.feature) if hasattr(X, "columns") else 0
        return self

    def predict(self, X):
        values = X.iloc[:, self.column_] if hasattr(X, "iloc") else X[:, self.column_]
        return np.asarray(values, dtype=float)


def _pipeline(estimator) -> Pipeline:
    # Scaler is inside the pipeline so it is refitted on each training fold --
    # fitting it once on the full series would leak future distribution info.
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


MODELS: dict[str, Callable[[], object]] = {
    "zero": ZeroForecast,
    "mean": MeanForecast,
    "persistence": PersistenceForecast,
    "ridge": lambda: _pipeline(Ridge(alpha=50.0)),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=300, max_depth=4, min_samples_leaf=50,
        max_features="sqrt", random_state=0, n_jobs=-1,
    ),
    "gradient_boosting": lambda: HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.02, max_iter=300,
        min_samples_leaf=50, l2_regularization=1.0, random_state=0,
    ),
}

#: Models that cost nothing and must be beaten before a result means anything.
BASELINE_MODELS = ("zero", "mean", "persistence")


def build_model(name: str):
    try:
        return MODELS[name]()
    except KeyError:
        raise ValueError(f"unknown model {name!r}; available: {', '.join(MODELS)}") from None


@dataclass
class WalkForwardResult:
    name: str
    predictions: pd.Series
    actuals: pd.Series
    metrics: ForecastMetrics
    fold_metrics: list[ForecastMetrics] = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        return len(self.fold_metrics)

    def as_dict(self) -> dict:
        return {"name": self.name, "n_folds": self.n_folds, **self.metrics.as_dict()}


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    model: str | Callable[[], object] = "ridge",
    *,
    train_size: int = 750,
    test_size: int = 126,
    expanding: bool = True,
    embargo: int = 1,
) -> WalkForwardResult:
    """Retrain on the past, predict the next block, repeat.

    Returns out-of-sample predictions for every test row across all folds --
    never a single fitted-on-everything fit. Raises if the series is too short
    to produce even one fold, rather than silently returning nothing.
    """
    name = model if isinstance(model, str) else getattr(model, "__name__", "custom")
    factory = (lambda: build_model(model)) if isinstance(model, str) else model

    folds: list[Fold] = list(
        walk_forward(
            len(X), train_size=train_size, test_size=test_size,
            expanding=expanding, embargo=embargo,
        )
    )
    if not folds:
        raise ValueError(
            f"{len(X)} rows is too few for walk-forward with train_size={train_size} "
            f"and test_size={test_size}; shorten the windows or load more history"
        )

    chunks: list[pd.Series] = []
    fold_metrics: list[ForecastMetrics] = []
    for fold in folds:
        X_train, y_train = X.iloc[fold.train], y.iloc[fold.train]
        X_test, y_test = X.iloc[fold.test], y.iloc[fold.test]

        estimator = factory()
        estimator.fit(X_train, y_train)
        predicted = pd.Series(
            np.asarray(estimator.predict(X_test), dtype=float), index=X_test.index
        )
        chunks.append(predicted)
        fold_metrics.append(forecast_metrics(y_test, predicted))

    predictions = pd.concat(chunks)
    actuals = y.loc[predictions.index]
    return WalkForwardResult(
        name=name,
        predictions=predictions,
        actuals=actuals,
        metrics=forecast_metrics(actuals, predictions),
        fold_metrics=fold_metrics,
    )


def compare_models(
    X: pd.DataFrame, y: pd.Series, names: list[str] | None = None, **kwargs
) -> dict[str, WalkForwardResult]:
    """Run several models over identical folds so the comparison is fair."""
    names = names or list(MODELS)
    return {name: walk_forward_predict(X, y, name, **kwargs) for name in names}
