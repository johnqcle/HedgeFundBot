"""Checks a written Kronos run against the exchange clock and the shadow rules.

Used by ``python -m tradingagents.forecasts --check``. It reads artifacts. It
does not call a model, a broker, or an LLM.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from tradingagents.forecasts.calendar import SessionCalendar
from tradingagents.forecasts.constants import (
    MODEL_ID,
    MODEL_REVISION,
    TOKENIZER_ID,
    TOKENIZER_REVISION,
)


def verify_forecast_dir(directory: Path) -> list[str]:
    """Empty means the artifact meets the shadow-forecast checks."""
    directory = Path(directory)
    forecast_path = directory / "forecast.json"
    observed_path = directory / "observed_ohlcv.csv"
    if not forecast_path.is_file():
        return ["forecast.json is missing"]
    forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
    problems: list[str] = []

    if forecast.get("status") != "ok":
        problems.append(f"status is {forecast.get('status')!r}, expected 'ok'")
    if forecast.get("influence") != "shadow":
        problems.append("influence is not shadow")
    if forecast.get("decision_influence") is not False:
        problems.append("decision_influence is not false")
    for flag in ("rating_changed", "stops_changed", "sizing_changed", "confidence_adjusted"):
        if forecast.get(flag) is not False:
            problems.append(f"{flag} is not false")
    model = forecast.get("model") or {}
    expected_model = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            problems.append(f"model.{key} is {model.get(key)!r}, expected {expected!r}")
    if model.get("weights_bundled") is not False:
        problems.append("weights_bundled is not false")
    if forecast.get("calendar_id") != "XNYS":
        problems.append(f"calendar_id is {forecast.get('calendar_id')!r}, expected 'XNYS'")
    if forecast.get("pretraining_contamination") != "unknown":
        problems.append("pretraining_contamination must stay unknown")
    amount = forecast.get("amount") or {}
    if amount.get("amount_is_approximation"):
        formula = amount.get("amount_approximation_formula") or ""
        if "mean(open, high, low, close)" not in formula:
            problems.append("an approximated amount is missing its formula")

    if not observed_path.is_file():
        problems.append("observed_ohlcv.csv is missing")
        return problems
    observed = pd.read_csv(observed_path)
    if "session_date" not in observed.columns:
        problems.append("observed_ohlcv.csv has no session_date")
        return problems

    try:
        calendar = SessionCalendar()
        asof = datetime.fromisoformat(forecast["asof"])
        trade_date = date.fromisoformat(forecast["trade_date"])
    except (KeyError, TypeError, ValueError) as exc:
        problems.append(f"asof or trade_date is unreadable ({exc})")
        return problems

    last: date | None = None
    observed_days: set[str] = set()
    for raw in observed["session_date"]:
        day = date.fromisoformat(str(raw)[:10])
        observed_days.add(day.isoformat())
        if not calendar.is_session(day):
            problems.append(f"observed {day.isoformat()} is not an XNYS session")
            continue
        if not calendar.bar_is_complete(day, asof):
            problems.append(f"observed {day.isoformat()} was not complete at the analysis as-of")
        if day > trade_date:
            problems.append(f"observed {day.isoformat()} is after the trade date")
        last = day

    for name, horizon in (forecast.get("horizons") or {}).items():
        if horizon.get("status") != "ok":
            problems.append(f"horizon {name} status is {horizon.get('status')!r}")
            continue
        if horizon.get("sample_count_passed_to_predict") != 1:
            problems.append(f"horizon {name} did not keep sample_count at 1")
        if horizon.get("mean_is_an_uncertainty_interval") is not False:
            problems.append(f"horizon {name} is labeled as an uncertainty interval")
        sessions = [date.fromisoformat(value) for value in horizon.get("sessions") or []]
        if last is None:
            problems.append(f"horizon {name} has no completed history to follow")
            continue
        expected = calendar.next_sessions(last, int(name))
        if sessions != expected:
            problems.append(
                f"horizon {name} sessions {[day.isoformat() for day in sessions]} "
                f"are not the next XNYS sessions {[day.isoformat() for day in expected]}"
            )
        for row in horizon.get("rows") or []:
            for column in ("open", "high", "low", "close"):
                value = row.get(column)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                    problems.append(f"horizon {name} {column} on {row.get('session')} is not a positive finite price")
            session = date.fromisoformat(row["session"])
            expected_close = calendar.session_close(session).isoformat()
            if row.get("session_close") != expected_close:
                problems.append(
                    f"horizon {name} close stamp {row.get('session_close')!r} "
                    f"is not the XNYS close {expected_close}"
                )
            if session.isoformat() in observed_days:
                problems.append(f"forecast session {session.isoformat()} is also in the observed file")
    return problems
