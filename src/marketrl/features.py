"""Causal technical features and the next-day target.

Every feature at row ``t`` is computed from information available at the close
of day ``t`` and no later. This is the single easiest thing to get wrong in a
price-prediction pipeline and the single most expensive: a centered rolling
window or a scaler fitted on the whole series produces a backtest that looks
excellent and loses money. :func:`assert_no_lookahead` mechanically checks the
property rather than trusting the code review, and the test suite runs it.

The target is the **next-day log return**, not the price. Predicting the price
level yields an R^2 near 1.0 that means nothing -- tomorrow's price is
approximately today's price. Returns are the part that is actually unknown.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "make_features",
    "make_dataset",
    "FEATURE_COLUMNS",
    "assert_no_lookahead",
    "TARGET",
]

TARGET = "target_next_log_return"


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder's smoothing, expressed as an EWM; strictly backward-looking.
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def _atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0.0, np.nan)


def make_features(prices: pd.DataFrame, *, price_column: str = "adj_close") -> pd.DataFrame:
    """Build the feature matrix from OHLCV bars.

    Features are deliberately scale-free (returns, z-scores, ratios) so a model
    trained on one price regime still applies after the stock doubles.
    """
    close = prices[price_column].astype(float)
    log_return = np.log(close).diff()
    features = pd.DataFrame(index=prices.index)

    # Recent returns: the raw momentum/reversal signal.
    for lag in (1, 2, 3, 5, 10):
        features[f"ret_lag_{lag}"] = log_return.shift(lag - 1)

    # Momentum over several horizons, volatility-normalized so the scale is
    # comparable across calm and turbulent regimes.
    for window in (5, 10, 21, 63):
        momentum = np.log(close / close.shift(window))
        vol = log_return.rolling(window).std() * np.sqrt(window)
        features[f"momentum_{window}"] = momentum / vol.replace(0.0, np.nan)

    # Realized volatility, and whether it is rising or falling.
    vol_21 = log_return.rolling(21).std()
    features["vol_5"] = log_return.rolling(5).std()
    features["vol_21"] = vol_21
    features["vol_ratio"] = log_return.rolling(5).std() / vol_21.replace(0.0, np.nan)

    # Mean-reversion: distance from moving averages, in standard deviations.
    for window in (10, 21, 63):
        features[f"ma_gap_{window}"] = _zscore(close, window)

    features["rsi_14"] = _rsi(close) / 100.0

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = (ema_12 - ema_26) / close
    features["macd"] = macd
    features["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()

    # Position within the recent range: 0 at the 1-year low, 1 at the high.
    window_high = prices["high"].rolling(252).max()
    window_low = prices["low"].rolling(252).min()
    span = (window_high - window_low).replace(0.0, np.nan)
    features["range_position_252"] = (close - window_low) / span

    features["atr_14"] = _atr(prices) / close
    features["intraday_range"] = (prices["high"] - prices["low"]) / close
    features["close_location"] = (
        (prices["close"] - prices["low"]) / (prices["high"] - prices["low"]).replace(0.0, np.nan)
    ).fillna(0.5)

    # Volume, as a z-score and against its own trend.
    volume = prices["volume"].astype(float).replace(0.0, np.nan)
    features["volume_z_21"] = _zscore(np.log(volume), 21)
    features["volume_trend"] = np.log(
        volume.rolling(5).mean() / volume.rolling(21).mean().replace(0.0, np.nan)
    )

    # Calendar effects, encoded cyclically so Friday and Monday stay adjacent.
    day_of_week = prices.index.dayofweek.to_numpy()
    features["dow_sin"] = np.sin(2 * np.pi * day_of_week / 5.0)
    features["dow_cos"] = np.cos(2 * np.pi * day_of_week / 5.0)

    return features.replace([np.inf, -np.inf], np.nan)


FEATURE_COLUMNS = list(
    make_features(
        pd.DataFrame(
            {
                "open": np.ones(300), "high": np.ones(300) * 1.01,
                "low": np.ones(300) * 0.99, "close": np.ones(300),
                "adj_close": np.ones(300), "volume": np.ones(300) * 1e6,
            },
            index=pd.bdate_range("2020-01-01", periods=300),
        )
    ).columns
)


def make_dataset(
    prices: pd.DataFrame, *, price_column: str = "adj_close", horizon: int = 1
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return ``(features, target, forward_return)`` aligned and NaN-free.

    ``target`` is the log return from the close of day ``t`` to the close of
    day ``t + horizon`` -- the quantity a model is asked to predict. It is
    returned separately from ``forward_return`` (numerically identical) to keep
    the roles distinct: one is what we fit, the other is what a position held
    from ``t`` actually earns in the backtest.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    close = prices[price_column].astype(float)
    features = make_features(prices, price_column=price_column)
    forward = np.log(close.shift(-horizon) / close).rename(TARGET)

    frame = features.join(forward).dropna()
    if frame.empty:
        raise ValueError(
            f"no usable rows after feature construction: {len(prices)} bars is too "
            f"few (need roughly 300 for the 252-day features to warm up)"
        )
    target = frame[TARGET]
    return frame[features.columns], target, target.copy()


def assert_no_lookahead(
    prices: pd.DataFrame, *, price_column: str = "adj_close", cut: int | None = None
) -> None:
    """Verify features at time t do not change when the future is rewritten.

    Recomputes the feature matrix on a truncated series and on a version whose
    post-``cut`` rows have been replaced with noise; any feature that differs on
    or before ``cut`` is reading the future. Raises :class:`AssertionError`
    naming the offending columns.
    """
    cut = cut if cut is not None else int(len(prices) * 0.7)
    baseline = make_features(prices.iloc[:cut], price_column=price_column)

    tampered = prices.copy()
    rng = np.random.default_rng(0)
    noise = rng.uniform(0.5, 2.0, size=len(tampered) - cut)
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        tampered.iloc[cut:, tampered.columns.get_loc(column)] *= noise
    perturbed = make_features(tampered, price_column=price_column).iloc[:cut]

    mismatched = [
        column
        for column in baseline.columns
        if not np.allclose(
            baseline[column].to_numpy(dtype=float),
            perturbed[column].to_numpy(dtype=float),
            equal_nan=True,
            rtol=1e-9,
            atol=1e-12,
        )
    ]
    if mismatched:
        raise AssertionError(
            f"lookahead detected: {mismatched} changed when only future rows were altered"
        )
