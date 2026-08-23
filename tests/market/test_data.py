import numpy as np
import pandas as pd
import pytest

from marketrl.data import (
    DataError,
    OHLCV_COLUMNS,
    load,
    load_csv,
    synthetic_prices,
)


def test_synthetic_is_deterministic_and_well_formed():
    a = synthetic_prices(400, seed=5)
    b = synthetic_prices(400, seed=5)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == OHLCV_COLUMNS
    assert len(a) == 400
    assert (a["close"] > 0).all()
    assert a.index.is_monotonic_increasing


def test_synthetic_respects_ohlc_ordering():
    frame = synthetic_prices(500, seed=2)
    assert (frame["high"] >= frame[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (frame["low"] <= frame[["open", "close"]].min(axis=1) + 1e-9).all()


def test_synthetic_is_always_flagged_as_synthetic():
    assert synthetic_prices(100).attrs["is_synthetic"] is True
    assert load("X", source="synthetic").attrs["is_synthetic"] is True


def test_synthetic_has_volatility_clustering():
    returns = np.log(synthetic_prices(3000, seed=9)["close"]).diff().dropna()
    # Squared returns should be autocorrelated -- the GARCH signature.
    squared = (returns**2).to_numpy()
    autocorr = np.corrcoef(squared[:-1], squared[1:])[0, 1]
    assert autocorr > 0.05


def test_signal_zero_is_genuinely_unpredictable():
    frame = synthetic_prices(4000, seed=4, signal=0.0)
    returns = np.log(frame["close"]).diff().dropna().to_numpy()
    assert abs(np.corrcoef(returns[:-1], returns[1:])[0, 1]) < 0.05


def test_planted_signal_is_actually_present():
    frame = synthetic_prices(4000, seed=4, signal=1.0)
    returns = np.log(frame["close"]).diff().dropna().to_numpy()
    assert abs(np.corrcoef(returns[:-1], returns[1:])[0, 1]) > 0.05


def test_network_failure_never_falls_back_to_synthetic(monkeypatch):
    import marketrl.data as data_module

    def blocked(url, timeout=30.0):
        raise DataError("blocked by egress policy")

    monkeypatch.setattr(data_module, "_http_get", blocked)
    with pytest.raises(DataError) as err:
        load("AAPL", source="auto", cache=False)
    # The error must name the remedy rather than quietly simulating prices.
    assert "synthetic" in str(err.value)
    assert "never a silent fallback" in str(err.value)


def test_load_csv_round_trip(tmp_path):
    original = synthetic_prices(300, seed=1)
    path = tmp_path / "bars.csv"
    original.to_csv(path)
    reloaded = load_csv(path, symbol="SYNTH")
    assert len(reloaded) == len(original)
    assert reloaded.attrs["is_synthetic"] is False  # a CSV is not synthetic
    np.testing.assert_allclose(reloaded["close"], original["close"])


def test_load_csv_accepts_the_adj_close_spelling(tmp_path):
    path = tmp_path / "bars.csv"
    path.write_text("Date,Close,Adj Close,Volume\n2024-01-02,10,9.5,100\n2024-01-03,11,10.5,110\n")
    frame = load_csv(path)
    assert frame["adj_close"].iloc[0] == 9.5


def test_load_csv_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(DataError, match="no date column"):
        load_csv(path)


def test_unusable_rows_are_dropped(tmp_path):
    path = tmp_path / "gaps.csv"
    path.write_text(
        "date,close\n2024-01-02,10\n2024-01-03,\n2024-01-04,-5\n2024-01-05,12\n"
    )
    frame = load_csv(path)
    assert len(frame) == 2  # the blank and the negative close are gone
