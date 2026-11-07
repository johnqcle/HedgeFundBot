"""Session clock for daily Kronos inputs.

The built-in calendar is weekdays in one timezone with a 09:30–16:00 regular
session. It does not know exchange holidays or early closes. Every manifest
records that limit. A daily bar is usable only after its session close, and
only when that close is at or before the analysis as-of.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from tradingagents.forecasts.constants import CALENDAR_ID, DEFAULT_TIMEZONE

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)

CALENDAR_LIMITATION = (
    "Weekdays only in the configured timezone. Exchange holidays and early "
    "closes are not modeled, so a forecast timestamp can fall on a closed "
    "session and a holiday with no bar is simply absent from the history. "
    "A daily bar is treated as complete at 16:00 local time."
)


@dataclass(frozen=True)
class SessionCalendar:
    """One named session clock. ``us_equity_weekday`` is the only built-in id."""

    timezone: str = DEFAULT_TIMEZONE
    calendar_id: str = CALENDAR_ID

    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def is_session(self, day: date) -> bool:
        return day.weekday() < 5

    def session_close(self, day: date) -> datetime:
        return datetime.combine(day, REGULAR_CLOSE, tzinfo=self.zone())

    def analysis_asof(self, trade_date: str, now: datetime) -> datetime:
        """Latest moment the run may know, given its trade date and the clock.

        The cutoff is the end of ``trade_date`` in the exchange timezone, and
        never later than ``now``. A historical trade date therefore cannot see
        later sessions. A same-day run cannot see a session that has not closed
        yet, because ``now`` is still before that close.
        """
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        day = datetime.strptime(trade_date, "%Y-%m-%d").date()
        end = datetime.combine(day, time(23, 59, 59), tzinfo=self.zone())
        return min(end, now.astimezone(self.zone()))

    def bar_is_complete(self, session_day: date, asof: datetime) -> bool:
        """Whether the daily bar for ``session_day`` had closed by ``asof``."""
        if not self.is_session(session_day):
            return False
        if asof.tzinfo is None:
            asof = asof.replace(tzinfo=UTC)
        return self.session_close(session_day) <= asof.astimezone(self.zone())

    def next_sessions(self, after: date, count: int) -> list[date]:
        """The next ``count`` weekday sessions strictly after ``after``."""
        if count < 1:
            raise ValueError("count must be positive")
        found: list[date] = []
        cursor = after
        while len(found) < count:
            cursor += timedelta(days=1)
            if self.is_session(cursor):
                found.append(cursor)
        return found

    def stamp(self, day: date) -> datetime:
        """Timestamp handed to Kronos: the session close, when the bar exists."""
        return self.session_close(day)


def require_calendar(calendar_id: str, timezone_name: str) -> SessionCalendar:
    """The built-in calendar, or an error the caller turns into an unavailable status."""
    if calendar_id != CALENDAR_ID:
        raise ValueError(
            f"unsupported kronos calendar {calendar_id!r}; the built-in calendar is {CALENDAR_ID}"
        )
    # ZoneInfo raises ZoneInfoNotFoundError on a bad name. That is a config error.
    ZoneInfo(timezone_name)
    return SessionCalendar(timezone=timezone_name, calendar_id=calendar_id)
