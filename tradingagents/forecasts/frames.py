"""Turn vendor bars into the frame Kronos actually sees.

Observations stay in their own columns. Forecast rows are never appended here.
Volume and amount follow the upstream optional-field rules without silently
inventing ``amount = volume * mean(OHLC)``, which is what
``KronosPredictor.predict`` does when volume is present and amount is not.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from tradingagents.forecasts.calendar import SessionCalendar
from tradingagents.forecasts.constants import AMOUNT_POLICY_APPROXIMATE, AMOUNT_POLICY_OMIT

PRICE_COLUMNS = ("open", "high", "low", "close")
UPSTREAM_COLUMNS = PRICE_COLUMNS + ("volume", "amount")


def _as_date(value) -> date | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    return ts.date()


def _valid_ohlc(row: pd.Series) -> bool:
    prices = [row[name] for name in PRICE_COLUMNS]
    if not all(np.isfinite(price) and price > 0 for price in prices):
        return False
    open_, high, low, close = prices
    return high >= max(open_, close, low) and low <= min(open_, close, high)


def completed_history(
    bars: pd.DataFrame,
    calendar: SessionCalendar,
    asof: datetime,
    *,
    lookback: int,
    min_bars: int,
) -> tuple[pd.DataFrame, dict]:
    """Completed daily bars at ``asof``, oldest first, capped at ``lookback``.

    Drops future sessions, sessions that have not closed, non-sessions, and
    rows whose OHLC is missing or impossible. Does not fill gaps. Duplicate
    session dates keep the last row and say so.
    """
    if "session_date" not in bars.columns:
        raise ValueError("bars require a session_date column")
    missing_price = [name for name in PRICE_COLUMNS if name not in bars.columns]
    if missing_price:
        raise ValueError(f"bars missing {missing_price}")

    frame = bars.copy()
    frame["session_date"] = frame["session_date"].map(_as_date)
    notes: list[str] = []
    dropped = {"undated": 0, "non_session": 0, "future_or_incomplete": 0, "invalid_ohlc": 0}

    keep_index: list = []
    for idx, row in frame.iterrows():
        session = row["session_date"]
        if session is None:
            dropped["undated"] += 1
            continue
        if not calendar.is_session(session):
            dropped["non_session"] += 1
            continue
        if not calendar.bar_is_complete(session, asof):
            dropped["future_or_incomplete"] += 1
            continue
        if not _valid_ohlc(row):
            dropped["invalid_ohlc"] += 1
            continue
        keep_index.append(idx)

    kept = frame.loc[keep_index].sort_values("session_date")
    duplicate_rows = int(kept["session_date"].duplicated(keep="last").sum())
    if duplicate_rows:
        notes.append(
            f"kept the last row of {duplicate_rows} duplicated session date(s); rows were not averaged"
        )
        kept = kept.drop_duplicates("session_date", keep="last")

    available = int(len(kept))
    if available > lookback:
        kept = kept.iloc[-lookback:]
    kept = kept.reset_index(drop=True)

    info = {
        "rows_in": int(len(bars)),
        "rows_completed": available,
        "rows_used": int(len(kept)),
        "lookback_requested": lookback,
        "lookback_used": int(len(kept)),
        "min_bars": min_bars,
        "dropped": dropped,
        "notes": notes,
        "first_session": kept["session_date"].iloc[0].isoformat() if len(kept) else None,
        "last_completed_session": kept["session_date"].iloc[-1].isoformat() if len(kept) else None,
    }
    # The floor is the completed history, not the lookback slice. A shorter
    # requested lookback is still a real window when enough sessions exist.
    if available < min_bars:
        info["rejection"] = "insufficient_completed_bars"
    elif available < lookback:
        info["notes"] = [
            *notes,
            f"history has {available} completed bars, shorter than lookback {lookback}",
        ]
    return kept, info


def model_inputs(bars: pd.DataFrame, amount_policy: str) -> tuple[pd.DataFrame, dict]:
    """Columns passed to ``KronosPredictor.predict``, plus how volume and amount were chosen.

    Upstream fills a missing volume and amount with zeros. If volume is present
    and amount is not, it sets ``amount = volume * mean(open, high, low, close)``
    with no flag. This wrapper never triggers that branch. An observed amount is
    passed through. The approximation is passed only when the policy asks for it,
    and the manifest names the formula.
    """
    if amount_policy not in (AMOUNT_POLICY_OMIT, AMOUNT_POLICY_APPROXIMATE):
        raise ValueError(f"unsupported amount policy {amount_policy!r}")

    prices = bars.loc[:, list(PRICE_COLUMNS)].reset_index(drop=True)
    meta: dict = {
        "amount_policy": amount_policy,
        "amount_source": "omitted",
        "volume_source": "omitted",
        "amount_is_approximation": False,
        "upstream_missing_field_behavior_avoided": (
            "volume without amount would make KronosPredictor set "
            "amount = volume * mean(open, high, low, close) without labeling it"
        ),
        "notes": [],
    }

    amount = _finite_series(bars, "amount")
    volume = _finite_series(bars, "volume")

    if amount is not None:
        out = prices.copy()
        out["amount"] = amount
        if volume is not None:
            out["volume"] = volume
            meta["volume_source"] = "observed"
        else:
            # Upstream requires both once one of them is present. Zero volume is
            # explicit because an amount was actually observed.
            out["volume"] = 0.0
            meta["volume_source"] = "filled_zero_amount_was_observed"
            meta["notes"].append("volume was missing or incomplete; passed as 0 because amount was observed")
        meta["amount_source"] = "observed"
        meta["columns_passed"] = list(out.columns)
        meta["upstream_fill"] = None
        return out, meta

    if amount_policy == AMOUNT_POLICY_APPROXIMATE and volume is not None:
        out = prices.copy()
        out["volume"] = volume
        out["amount"] = volume.to_numpy(dtype=float) * prices.mean(axis=1).to_numpy(dtype=float)
        meta.update(
            volume_source="observed",
            amount_source=AMOUNT_POLICY_APPROXIMATE,
            amount_is_approximation=True,
            amount_approximation_formula="volume * mean(open, high, low, close)",
            columns_passed=list(out.columns),
            upstream_fill=None,
        )
        meta["notes"].append(
            "amount is an approximation, not an observed turnover; it is labeled and was not filled inside KronosPredictor"
        )
        return out, meta

    if volume is not None:
        meta["volume_source"] = "omitted_to_avoid_synthetic_amount"
        meta["notes"].append(
            "share volume was observed and is in the audit file; it was not passed to Kronos "
            "because amount was not observed and the omit policy refuses a synthetic amount"
        )
    else:
        meta["volume_source"] = "not_in_source"
    meta["upstream_fill"] = "predict() will set the omitted volume and amount to 0"
    meta["columns_passed"] = list(PRICE_COLUMNS)
    return prices, meta


def _finite_series(bars: pd.DataFrame, name: str) -> pd.Series | None:
    """A column only when every row has a finite value. Gaps are not filled."""
    if name not in bars.columns:
        return None
    series = pd.to_numeric(bars[name], errors="coerce").reset_index(drop=True)
    if series.isna().any() or not np.isfinite(series.to_numpy(dtype=float)).all():
        return None
    return series


def history_stamps(session_dates, calendar: SessionCalendar) -> pd.Series:
    """Session-close timestamps, one per history row, as a Series Kronos can read with ``.dt``."""
    stamps = [calendar.stamp(day) for day in session_dates]
    return pd.Series(pd.to_datetime(stamps))


def future_stamps(last_session: date, horizon: int, calendar: SessionCalendar) -> tuple[list[date], pd.Series]:
    """Future session dates and their close timestamps. These are not observations."""
    days = calendar.next_sessions(last_session, horizon)
    return days, history_stamps(days, calendar)
