"""XNYS sessions, holidays, and early closes. No network and no model."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from tradingagents.forecasts.calendar import (
    CalendarUnavailable,
    SessionCalendar,
    latest_completed_session,
    require_calendar,
)
from tradingagents.forecasts.constants import EXCHANGE_CALENDARS_VERSION
from tradingagents.forecasts.frames import completed_history

NY = ZoneInfo("America/New_York")


@pytest.mark.unit
def test_thanksgiving_week_skips_the_holiday_and_closes_friday_early():
    calendar = SessionCalendar()
    assert calendar.library_version == EXCHANGE_CALENDARS_VERSION
    assert calendar.is_session(date(2025, 11, 26))
    assert not calendar.is_session(date(2025, 11, 27))
    assert calendar.is_session(date(2025, 11, 28))
    assert [day.isoformat() for day in calendar.next_sessions(date(2025, 11, 26), 4)] == [
        "2025-11-28", "2025-12-01", "2025-12-02", "2025-12-03",
    ]
    friday = calendar.session_close(date(2025, 11, 28))
    wednesday = calendar.session_close(date(2025, 11, 26))
    assert friday.hour == 13 and friday.minute == 0 and friday.utcoffset().total_seconds() == -5 * 3600
    assert wednesday.hour == 16 and wednesday.minute == 0


@pytest.mark.unit
def test_an_early_close_is_complete_at_1300_and_not_before():
    calendar = SessionCalendar()
    day = date(2025, 11, 28)
    before = datetime(2025, 11, 28, 12, 59, tzinfo=NY)
    at_close = datetime(2025, 11, 28, 13, 0, tzinfo=NY)
    assert not calendar.bar_is_complete(day, before)
    assert calendar.bar_is_complete(day, at_close)
    # A fixed 16:00 rule would still call this bar incomplete at 13:00.
    assert calendar.session_close(day) <= at_close


@pytest.mark.unit
def test_july_third_and_christmas_eve_are_early_and_the_holidays_are_closed():
    calendar = SessionCalendar()
    assert calendar.session_close(date(2025, 7, 3)).hour == 13
    assert not calendar.is_session(date(2025, 7, 4))
    assert calendar.session_close(date(2025, 12, 24)).hour == 13
    assert not calendar.is_session(date(2025, 12, 25))
    assert calendar.next_sessions(date(2025, 12, 24), 1) == [date(2025, 12, 26)]


@pytest.mark.unit
def test_history_drops_a_holiday_row_and_an_unfinished_early_close():
    calendar = SessionCalendar()
    days = [date(2025, 11, day) for day in (26, 27, 28)]
    bars = pd.DataFrame({
        "session_date": days,
        "open": [10, 11, 12],
        "high": [11, 12, 13],
        "low": [9, 10, 11],
        "close": [10.5, 11.5, 12.5],
    })
    asof = calendar.analysis_asof("2025-11-28", datetime(2025, 11, 28, 12, 30, tzinfo=NY))
    kept, info = completed_history(bars, calendar, asof, lookback=10, min_bars=1)
    assert [day.isoformat() for day in kept["session_date"]] == ["2025-11-26"]
    assert info["dropped"]["non_session"] == 1
    assert info["dropped"]["future_or_incomplete"] == 1

    later = calendar.analysis_asof("2025-11-28", datetime(2025, 11, 28, 13, 30, tzinfo=NY))
    kept_later, _info = completed_history(bars, calendar, later, lookback=10, min_bars=1)
    assert [day.isoformat() for day in kept_later["session_date"]] == ["2025-11-26", "2025-11-28"]


@pytest.mark.unit
def test_a_historical_asof_does_not_take_the_session_after_a_holiday():
    calendar = SessionCalendar()
    days = [date(2025, 11, day) for day in (26, 27, 28)]
    bars = pd.DataFrame({
        "session_date": days,
        "open": [10, 11, 12], "high": [11, 12, 13], "low": [9, 10, 11], "close": [10.5, 11.5, 12.5],
    })
    asof = calendar.analysis_asof("2025-11-26", datetime(2025, 12, 15, 12, 0, tzinfo=NY))
    kept, _info = completed_history(bars, calendar, asof, lookback=10, min_bars=1)
    assert [day.isoformat() for day in kept["session_date"]] == ["2025-11-26"]
    assert asof.date().isoformat() == "2025-11-26"


@pytest.mark.unit
def test_nasdaq_and_nyse_share_this_schedules_reviewed_window():
    import exchange_calendars as xcals

    nyse = xcals.get_calendar("XNYS")
    nasdaq = xcals.get_calendar("XNAS")
    start, end = "2024-01-01", "2026-12-31"
    nyse_sessions = nyse.sessions_in_range(start, end)
    nasdaq_sessions = nasdaq.sessions_in_range(start, end)
    assert nyse_sessions.equals(nasdaq_sessions)
    mismatches = [
        session for session in nyse_sessions
        if nyse.session_close(session) != nasdaq.session_close(session)
        or nyse.session_open(session) != nasdaq.session_open(session)
    ]
    assert mismatches == []


@pytest.mark.unit
def test_weekday_calendar_and_a_foreign_timezone_are_refused():
    with pytest.raises(CalendarUnavailable) as weekday:
        require_calendar("us_equity_weekday", "America/New_York")
    assert weekday.value.reason == "unsupported_calendar"
    with pytest.raises(CalendarUnavailable) as zone:
        require_calendar("XNYS", "UTC")
    assert zone.value.reason == "calendar_timezone_mismatch"


@pytest.mark.unit
def test_latest_completed_session_on_a_saturday_is_friday():
    # 2026-11-07 is a Saturday. 2026-11-06 is a full XNYS session.
    found = latest_completed_session(datetime(2026, 11, 7, 15, 0, tzinfo=NY))
    assert found == date(2026, 11, 6)
    before_the_close = latest_completed_session(datetime(2026, 11, 6, 15, 30, tzinfo=NY))
    assert before_the_close == date(2026, 11, 5)


@pytest.mark.unit
def test_the_forecast_command_does_not_construct_the_agent_graph(monkeypatch, tmp_path):
    from tradingagents.forecasts import __main__ as entry
    from tradingagents.forecasts.runner import KronosEvidence

    source = Path(entry.__file__).read_text(encoding="utf-8")
    assert "TradingAgentsGraph" not in source
    assert "XAI_API" not in source
    assert "langchain" not in source

    seen = []
    monkeypatch.setattr(entry, "latest_completed_session", lambda now=None: date(2026, 11, 6))

    def fake(**kwargs):
        seen.append(kwargs)
        return KronosEvidence("ok", "shadow\n", "", False)

    monkeypatch.setattr(entry, "run_kronos_evidence", fake)
    code = entry.main([
        "--ticker", "AVGO", "--ticker", "UI", "--latest-completed",
        "--results-dir", str(tmp_path),
    ])
    assert code == 0
    assert [call["ticker"] for call in seen] == ["AVGO", "UI"]
    assert {call["trade_date"] for call in seen} == {"2026-11-06"}
    assert all(call["config"]["kronos_influence"] == "shadow" for call in seen)
    assert all(call["config"]["kronos_enabled"] is True for call in seen)
