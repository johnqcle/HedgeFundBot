"""NYSE cash-session clock for Kronos inputs.

Future timestamps and bar completion both use the XNYS schedule from
``exchange-calendars``: holidays are not sessions, and an early close is
complete at that day's close (13:00 America/New_York), not at 16:00.
NASDAQ cash equities use the same open, close, and holiday set in the
reviewed library version, so AVGO and UI share this clock.

A weekday calendar is not a fallback. If the library is missing or the date
is outside its schedule, the run is unavailable and no sessions are invented.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd

from tradingagents.forecasts.constants import EXCHANGE_CALENDARS_VERSION

CALENDAR_ID = "XNYS"
EXCHANGE_TIMEZONE = "America/New_York"
_ZONE = ZoneInfo(EXCHANGE_TIMEZONE)

CALENDAR_LIMITATION = (
    "XNYS sessions from exchange-calendars, including holidays and early closes. "
    "A daily bar is complete at that session's close in America/New_York "
    "(16:00, or 13:00 when the schedule says the session closes early). "
    "Forecast timestamps are those closes and are not observations. "
    f"Reviewed with exchange-calendars {EXCHANGE_CALENDARS_VERSION}, in which "
    "XNAS has the same sessions and open/close times as XNYS from 2024-01-01 "
    "through 2026-12-31. The library schedule does not extend forever; a date "
    "outside it is unavailable rather than filled with weekdays."
)


class CalendarUnavailable(Exception):
    """The exchange schedule cannot be applied. No weekday substitute is used."""

    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


@lru_cache(maxsize=1)
def _load_xnys():
    try:
        import exchange_calendars as xcals
    except ImportError as exc:
        raise CalendarUnavailable(
            "exchange_calendar_missing",
            'US sessions need the exchange-calendars package. Install it with '
            'pip install -e ".[kronos]" in the HedgeFundBot environment. '
            "Weekdays are not used in its place.",
        ) from exc
    return xcals.get_calendar(CALENDAR_ID), xcals.__version__


class SessionCalendar:
    """XNYS schedule. ``timezone`` is the exchange zone, not a caller override."""

    def __init__(self, timezone: str = EXCHANGE_TIMEZONE, calendar_id: str = CALENDAR_ID):
        if calendar_id == "us_equity_weekday":
            raise CalendarUnavailable(
                "unsupported_calendar",
                "us_equity_weekday was removed because it places forecast timestamps "
                "on exchange holidays and treats early closes as open until 16:00. "
                "Set TRADINGAGENTS_KRONOS_CALENDAR=XNYS.",
            )
        if calendar_id != CALENDAR_ID:
            raise CalendarUnavailable(
                "unsupported_calendar",
                f"unsupported kronos calendar {calendar_id!r}. The supported calendar is {CALENDAR_ID}.",
            )
        if timezone != EXCHANGE_TIMEZONE:
            raise CalendarUnavailable(
                "calendar_timezone_mismatch",
                f"XNYS sessions are {EXCHANGE_TIMEZONE}. Refusing timezone {timezone!r} "
                "so a close is not stamped in another zone.",
            )
        self._cal, self.library_version = _load_xnys()
        self.timezone = EXCHANGE_TIMEZONE
        self.calendar_id = CALENDAR_ID
        self.first_session = self._cal.first_session.date()
        self.last_session = self._cal.last_session.date()

    def zone(self) -> ZoneInfo:
        return _ZONE

    def covers(self, day: date) -> bool:
        return self.first_session <= day <= self.last_session

    def is_session(self, day: date) -> bool:
        if not self.covers(day):
            return False
        return bool(self._cal.is_session(pd.Timestamp(day)))

    def session_close(self, day: date) -> datetime:
        """Exchange-local close. Early-close days are 13:00, not 16:00."""
        if not self.is_session(day):
            raise CalendarUnavailable(
                "calendar_out_of_range",
                f"{day.isoformat()} is not an XNYS session.",
            )
        closed = self._cal.session_close(pd.Timestamp(day)).tz_convert(EXCHANGE_TIMEZONE)
        return closed.to_pydatetime()

    def analysis_asof(self, trade_date: str, now: datetime) -> datetime:
        """Latest moment the run may know: end of the trade date, and not after ``now``."""
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        day = datetime.strptime(trade_date, "%Y-%m-%d").date()
        end = datetime.combine(day, time(23, 59, 59), tzinfo=self.zone())
        return min(end, now.astimezone(self.zone()))

    def bar_is_complete(self, session_day: date, asof: datetime) -> bool:
        """Whether the daily bar had closed by ``asof``. Holidays are never complete."""
        if not self.is_session(session_day):
            return False
        if asof.tzinfo is None:
            asof = asof.replace(tzinfo=UTC)
        return self.session_close(session_day) <= asof.astimezone(self.zone())

    def next_sessions(self, after: date, count: int) -> list[date]:
        """The next ``count`` XNYS sessions strictly after ``after``."""
        if count < 1:
            raise ValueError("count must be positive")
        if not self.covers(after) and after > self.last_session:
            raise CalendarUnavailable(
                "calendar_out_of_range",
                f"{after.isoformat()} is after the XNYS schedule, which ends {self.last_session.isoformat()} "
                f"in exchange-calendars {self.library_version}.",
            )
        try:
            ts = pd.Timestamp(after)
            start = self._cal.next_session(ts) if self.is_session(after) else self._cal.date_to_session(
                ts, direction="next"
            )
            window = self._cal.sessions_window(start, count)
        except CalendarUnavailable:
            raise
        except Exception as exc:
            if type(exc).__name__ in {"DateOutOfBounds", "RequestedSessionOutOfBounds"}:
                raise CalendarUnavailable(
                    "calendar_out_of_range",
                    "The requested sessions run past the XNYS schedule in "
                    f"exchange-calendars {self.library_version} "
                    f"({self.first_session.isoformat()} through {self.last_session.isoformat()}). {exc}",
                ) from exc
            raise
        return [pd.Timestamp(day).date() for day in window]

    def stamp(self, day: date) -> datetime:
        """Timestamp handed to Kronos: the session close, in exchange-local time."""
        return self.session_close(day)

    @property
    def limitation(self) -> str:
        reviewed = ""
        if self.library_version != EXCHANGE_CALENDARS_VERSION:
            reviewed = (
                f" This environment has exchange-calendars {self.library_version}; "
                f"the reviewed pin is {EXCHANGE_CALENDARS_VERSION}."
            )
        return CALENDAR_LIMITATION + reviewed


def require_calendar(calendar_id: str, timezone_name: str) -> SessionCalendar:
    """The XNYS calendar, or an error the caller records as unavailable."""
    return SessionCalendar(timezone=timezone_name, calendar_id=calendar_id)


def latest_completed_session(now: datetime | None = None) -> date:
    """The most recent XNYS session whose close is at or before ``now``."""
    calendar = require_calendar(CALENDAR_ID, EXCHANGE_TIMEZONE)
    clock = now or datetime.now(UTC)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=UTC)
    local_today = clock.astimezone(calendar.zone()).date()
    # Ten calendar days covers a long holiday weekend without walking the whole schedule.
    for offset in range(0, 14):
        day = date.fromordinal(local_today.toordinal() - offset)
        if calendar.covers(day) and calendar.bar_is_complete(day, clock):
            return day
    raise CalendarUnavailable(
        "calendar_out_of_range",
        f"No completed XNYS session in the 14 days through {local_today.isoformat()}.",
    )
