"""Daily bars for Kronos, on a split-consistent OHLC and volume basis.

The indicator path uses ``auto_adjust=True``, which rewrites Open/High/Low/Close
by the Adj Close ratio and leaves volume untouched (yfinance ``utils.auto_adjust``).
That makes dividend-adjusted prices sit next to share volume. Kronos gets the
other series: ``auto_adjust=False``, so Open/High/Low/Close stay on Yahoo's
split-adjusted close and volume stays the share volume published with that
close. Adj Close is kept for the audit file and is not a model input.

The download end is the day after the analysis date, exclusive, so a backtest
does not request later sessions. Rows dated after the analysis date are dropped
again here. Whether the analysis date's own bar has closed is decided by the
session calendar in the forecast layer, which also drops an incomplete bar.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import NoMarketDataError, VendorUnavailableError
from tradingagents.dataflows.files import replace_file
from tradingagents.dataflows.symbols import normalize_symbol, safe_ticker_component
from tradingagents.dataflows.vendors.yahoo.common import yf_retry
from tradingagents.forecasts.calendar import SessionCalendar
from tradingagents.forecasts.constants import DEFAULT_TIMEZONE

_PRICE = ("open", "high", "low", "close")

ADJUSTMENT = {
    "vendor": "yfinance",
    "auto_adjust": False,
    "policy": "yahoo_split_adjusted_ohlc_and_volume",
    "ohlc": (
        "Yahoo Close with auto_adjust=False is split-adjusted and not dividend-adjusted. "
        "Open, High, and Low from that response are on the same basis."
    ),
    "volume": "Share volume as returned with auto_adjust=False. yfinance does not dividend-adjust volume.",
    "adj_close": "Retained in the observed audit file only. Not passed to Kronos.",
    "amount": "Yahoo daily bars do not include a turnover amount.",
    "consistent": True,
    "consistent_means": "OHLC and volume share a split adjustment. Dividend adjustment is not applied.",
    "differs_from_indicator_feed": (
        "load_ohlcv uses auto_adjust=True for indicator continuity. This series does not, "
        "so a Kronos input is not the dividend-adjusted indicator series."
    ),
}


def load_split_consistent_daily_bars(
    symbol: str,
    as_of_date: str,
    *,
    now: datetime | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Completed-through-``as_of_date`` daily rows and an adjustment description.

    ``session_date`` is the exchange-local date of the bar. Timezone-aware
    timestamps are converted to America/New_York. Naive timestamps keep their
    wall-clock date.
    """
    canonical = normalize_symbol(symbol)
    safe_symbol = safe_ticker_component(canonical)
    as_of = pd.Timestamp(as_of_date).normalize()
    clock = now or datetime.now(UTC)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=UTC)

    config = get_config()
    cache_dir = config["data_cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{safe_symbol}-YFin-kronos-split-{as_of.strftime('%Y-%m-%d')}.csv")

    # yfinance end is exclusive. Cap it at the day after the analysis date so
    # the request itself does not ask for later sessions.
    start = (as_of - pd.DateOffset(years=12)).strftime("%Y-%m-%d")
    end = (as_of + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    frame = _read_cache(cache_path, as_of_date, clock)
    if frame is None:
        downloaded = yf_retry(lambda: yf.Ticker(canonical).history(
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            actions=False,
        ))
        if downloaded is None or downloaded.empty:
            raise NoMarketDataError(symbol, canonical, "no Kronos price rows")
        frame = _normalize_history(downloaded)
        if frame.empty:
            raise NoMarketDataError(symbol, canonical, "no Kronos price rows")
        replace_file(cache_path, lambda temp: frame.to_csv(temp, index=False))

    before = len(frame)
    frame = frame[frame["session_date"] <= as_of.date()].copy()
    frame = frame.sort_values("session_date").reset_index(drop=True)
    if frame.empty:
        raise NoMarketDataError(symbol, canonical, "no Kronos rows on or before the analysis date")

    meta = {
        "vendor": "yfinance",
        "symbol": symbol,
        "canonical": canonical,
        "interval": "1d",
        "download_start": start,
        "download_end_exclusive": end,
        "rows_downloaded": before,
        "rows_on_or_before_trade_date": int(len(frame)),
        "session_date_rule": (
            f"timezone-aware timestamps are converted to {DEFAULT_TIMEZONE}; "
            "naive timestamps keep their wall-clock date"
        ),
        "adjustment": dict(ADJUSTMENT),
    }
    return frame, meta


def _read_cache(path: str, as_of_date: str, now: datetime) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    calendar = SessionCalendar()
    written = datetime.fromtimestamp(os.path.getmtime(path), tz=calendar.zone())
    try:
        trade_day = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    except ValueError:
        return None
    asof = calendar.analysis_asof(as_of_date, now)
    # A file written before the session close can hold a partial daily bar.
    # Once that session is complete, refuse the early file.
    if calendar.bar_is_complete(trade_day, asof) and written < calendar.session_close(trade_day):
        return None
    try:
        cached = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeError):
        return None
    if cached.empty or "close" not in cached.columns or "session_date" not in cached.columns:
        return None
    cached["session_date"] = pd.to_datetime(cached["session_date"], errors="coerce").dt.date
    return cached


def _normalize_history(downloaded: pd.DataFrame) -> pd.DataFrame:
    data = downloaded.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.reset_index()
    date_column = _date_column(data)
    if date_column is None:
        raise VendorUnavailableError("Yahoo history had no date column")
    renamed = {
        date_column: "bar_time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    data = data.rename(columns={key: value for key, value in renamed.items() if key in data.columns})
    missing = [name for name in _PRICE if name not in data.columns]
    if missing:
        raise VendorUnavailableError(f"Yahoo history missing {missing}")
    data["session_date"] = data["bar_time"].map(_session_date)
    keep = ["session_date", *_PRICE]
    for optional in ("volume", "adj_close"):
        if optional in data.columns:
            keep.append(optional)
    out = data.loc[:, keep].dropna(subset=["session_date"]).copy()
    for name in _PRICE:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    if "adj_close" in out.columns:
        out["adj_close"] = pd.to_numeric(out["adj_close"], errors="coerce")
    return out.reset_index(drop=True)


def _date_column(data: pd.DataFrame) -> str | None:
    for name in ("Date", "Datetime", "date", "index"):
        if name in data.columns:
            return name
    return None


def _session_date(value):
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert(DEFAULT_TIMEZONE)
    return ts.date()
