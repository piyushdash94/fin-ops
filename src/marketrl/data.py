"""Price data loading: Yahoo Finance, Stooq, local CSV, or synthetic.

Design rule that matters more than any other here: **synthetic data is never a
silent fallback**. If a network source fails, loading raises. You only get
simulated prices by asking for them by name, and every frame carries an
``attrs["is_synthetic"]`` flag that the reporting layer prints in red ink.
Backtest results quietly computed on made-up prices are worse than no results.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "OHLCV_COLUMNS",
    "DataError",
    "load",
    "load_yahoo",
    "load_stooq",
    "load_csv",
    "synthetic_prices",
    "cache_dir",
]

OHLCV_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class DataError(RuntimeError):
    """Raised when price data cannot be obtained from the requested source."""


def cache_dir() -> Path:
    path = Path(os.environ.get("MARKETRL_HOME", Path.home() / ".cache" / "marketrl"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _http_get(url: str, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as err:
        raise DataError(f"HTTP {err.code} from {urllib.parse.urlsplit(url).netloc}") from err
    except Exception as err:  # URLError, timeouts, proxy denials
        raise DataError(f"could not reach {urllib.parse.urlsplit(url).netloc}: {err}") from err


def _finalize(frame: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
    """Normalize, sort, drop unusable rows and attach provenance."""
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame.index.name = "date"
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    for column in OHLCV_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[OHLCV_COLUMNS]

    # A row with no close is unusable; a non-positive close breaks log returns.
    frame = frame[frame["close"].notna() & (frame["close"] > 0)]
    frame["adj_close"] = frame["adj_close"].fillna(frame["close"])
    frame["volume"] = frame["volume"].fillna(0.0)

    if frame.empty:
        raise DataError(f"no usable rows for {symbol} from {source}")

    frame.attrs.update(symbol=symbol, source=source, is_synthetic=(source == "synthetic"))
    return frame


def load_yahoo(symbol: str, start: str | dt.date | None = None,
               end: str | dt.date | None = None, *, interval: str = "1d") -> pd.DataFrame:
    """Load daily bars from Yahoo Finance's public chart endpoint.

    Uses the JSON chart API directly rather than depending on ``yfinance`` --
    one fewer dependency, and the failure modes are legible. ``adj_close`` is
    the split- and dividend-adjusted series, which is the one to model.
    """
    period1 = int(pd.Timestamp(start or "1990-01-01").timestamp())
    period2 = int(pd.Timestamp(end or dt.date.today() + dt.timedelta(days=1)).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}"
        f"?period1={period1}&period2={period2}&interval={interval}"
        f"&events=div%2Csplit&includeAdjustedClose=true"
    )
    payload = json.loads(_http_get(url))

    error = (payload.get("chart") or {}).get("error")
    if error:
        raise DataError(f"Yahoo rejected {symbol}: {error}")
    results = (payload.get("chart") or {}).get("result") or []
    if not results:
        raise DataError(f"Yahoo returned no result for {symbol}")

    result = results[0]
    timestamps = result.get("timestamp") or []
    if not timestamps:
        raise DataError(f"Yahoo returned no bars for {symbol} in that date range")

    quote = result["indicators"]["quote"][0]
    adj = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    frame = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "adj_close": adj if adj is not None else quote.get("close"),
            "volume": quote.get("volume"),
        },
        index=pd.to_datetime(timestamps, unit="s"),
    )
    return _finalize(frame, symbol, "yahoo")


def load_stooq(symbol: str, start=None, end=None) -> pd.DataFrame:
    """Load daily bars from Stooq's free CSV endpoint.

    A useful second source: no key, no rate limit worth worrying about. Stooq
    symbols carry a market suffix (``aapl.us``), added here when absent. Stooq
    publishes adjusted prices, so ``close`` and ``adj_close`` are the same.
    """
    ticker = symbol.lower()
    if "." not in ticker:
        ticker = f"{ticker}.us"
    url = f"https://stooq.com/q/d/l/?s={urllib.parse.quote(ticker)}&i=d"
    raw = _http_get(url).decode("utf-8", errors="replace")
    if not raw.lower().startswith("date"):
        raise DataError(f"Stooq returned no data for {symbol} (got {raw[:60]!r})")

    frame = pd.read_csv(io.StringIO(raw))
    frame.columns = [c.strip().lower() for c in frame.columns]
    frame = frame.set_index("date")
    frame["adj_close"] = frame.get("close")
    frame = _finalize(frame, symbol, "stooq")
    return _slice(frame, start, end)


def load_csv(path: str | Path, symbol: str | None = None) -> pd.DataFrame:
    """Load bars from a local CSV (any open dataset exported to disk).

    Expects a date column plus ``close``; open/high/low/volume/adj_close are
    optional. Column names are matched case-insensitively and tolerate the
    common ``Adj Close`` spelling.
    """
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]

    date_col = next((c for c in ("date", "datetime", "timestamp", "time") if c in frame), None)
    if date_col is None:
        raise DataError(f"{path}: no date column found (looked for date/datetime/timestamp)")
    if "close" not in frame:
        raise DataError(f"{path}: no close column found")

    frame = frame.set_index(date_col)
    if "adjclose" in frame and "adj_close" not in frame:
        frame["adj_close"] = frame["adjclose"]
    return _finalize(frame, symbol or Path(path).stem.upper(), "csv")


def _slice(frame: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    attrs = dict(frame.attrs)
    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame.index <= pd.Timestamp(end)]
    frame.attrs.update(attrs)
    return frame


def synthetic_prices(
    n_days: int = 2500,
    *,
    seed: int = 0,
    signal: float = 0.25,
    annual_drift: float = 0.07,
    start: str = "2015-01-01",
    symbol: str = "SYNTH",
) -> pd.DataFrame:
    """Generate a simulated price series with known, tunable predictability.

    Returns follow a GARCH(1,1) volatility process -- so the series has the fat
    tails and volatility clustering real equities show -- plus an AR(1)
    predictable component whose strength is set by ``signal``:

        r_t = drift + signal * phi * z_{t-1} + sigma_t * e_t

    ``signal=0`` produces an efficient, genuinely unpredictable market. That is
    the important setting: it lets the test suite assert that the pipeline
    *fails to find* an edge that isn't there, which is the failure mode that
    matters in backtesting. Higher values plant a real edge that a working
    pipeline must be able to recover.
    """
    rng = np.random.default_rng(seed)

    omega, alpha, beta = 2e-6, 0.09, 0.88
    daily_drift = annual_drift / 252.0
    sigma2 = omega / max(1e-9, 1 - alpha - beta)

    returns = np.zeros(n_days)
    shock = 0.0
    prev_z = 0.0
    for t in range(n_days):
        sigma2 = omega + alpha * shock**2 + beta * sigma2
        sigma = np.sqrt(sigma2)
        eps = rng.standard_normal()
        shock = sigma * eps
        # Mean-reverting predictable component driven by the previous shock.
        returns[t] = daily_drift - signal * 0.05 * prev_z + shock
        prev_z = shock / max(sigma, 1e-9)

    close = 100.0 * np.exp(np.cumsum(returns))
    # Plausible intraday structure around each close.
    spread = np.abs(rng.standard_normal(n_days)) * 0.004 + 0.001
    open_ = close * np.exp(rng.standard_normal(n_days) * 0.002)
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    # Volume rises with volatility, as it does in real markets.
    volume = np.exp(15 + 0.8 * np.abs(returns) / max(returns.std(), 1e-9) * 0.3
                    + rng.standard_normal(n_days) * 0.25)

    index = pd.bdate_range(start=start, periods=n_days)
    frame = pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "adj_close": close, "volume": volume.round(),
        },
        index=index,
    )
    frame = _finalize(frame, symbol, "synthetic")
    frame.attrs.update(signal=signal, seed=seed)
    return frame


def load(
    symbol: str = "AAPL",
    start: str | dt.date | None = "2015-01-01",
    end: str | dt.date | None = None,
    *,
    source: str = "auto",
    cache: bool = True,
    max_age_hours: float = 12.0,
) -> pd.DataFrame:
    """Load daily bars, with an on-disk cache.

    ``source`` is one of ``auto`` (Yahoo, then Stooq), ``yahoo``, ``stooq``,
    ``synthetic``, or a path to a CSV file. Network failure raises
    :class:`DataError` -- it never degrades to synthetic data silently.
    """
    if source == "synthetic":
        return synthetic_prices(symbol=symbol)
    if source not in {"auto", "yahoo", "stooq"}:
        return _slice(load_csv(source, symbol=symbol), start, end)

    key = f"{symbol.upper().replace('/', '_')}_{source}.csv"
    path = cache_dir() / key
    if cache and path.exists():
        age_hours = (dt.datetime.now().timestamp() - path.stat().st_mtime) / 3600
        if age_hours < max_age_hours:
            frame = load_csv(path, symbol=symbol)
            frame.attrs["source"] = f"{source} (cached)"
            return _slice(frame, start, end)

    attempts = ["yahoo", "stooq"] if source == "auto" else [source]
    failures = []
    for name in attempts:
        try:
            frame = load_yahoo(symbol, start, end) if name == "yahoo" else load_stooq(symbol, start, end)
        except DataError as err:
            failures.append(f"{name}: {err}")
            continue
        if cache:
            try:
                frame.to_csv(path)
            except OSError:
                pass
        return _slice(frame, start, end)

    raise DataError(
        f"could not load {symbol} from {' or '.join(attempts)}.\n  "
        + "\n  ".join(failures)
        + "\nPass source='<path.csv>' to use a local file, or source='synthetic' "
          "for simulated prices (clearly labelled, never a silent fallback)."
    )
