from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class DateWindow:
    label: str
    from_iso: str
    to_iso: str


def today_yesterday_windows(now: datetime | None = None, tz: ZoneInfo | None = None) -> list[DateWindow]:
    tz = tz or ZoneInfo("Europe/London")
    local_today = _local_today(now=now, tz=tz)
    return date_range_windows(local_today - timedelta(days=1), local_today, tz=tz)


def last_n_days_windows(days: int, now: datetime | None = None, tz: ZoneInfo | None = None) -> list[DateWindow]:
    if days < 1:
        raise ValueError("days must be at least 1")
    tz = tz or ZoneInfo("Europe/London")
    local_today = _local_today(now=now, tz=tz)
    start_day = local_today - timedelta(days=days - 1)
    return date_range_windows(start_day, local_today, tz=tz)


def date_range_windows(start_date: date, end_date: date, tz: ZoneInfo | None = None) -> list[DateWindow]:
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")
    tz = tz or ZoneInfo("Europe/London")
    day_count = (end_date - start_date).days + 1
    return [_window_for_day(start_date + timedelta(days=offset), tz) for offset in range(day_count)]


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _local_today(now: datetime | None, tz: ZoneInfo) -> date:
    return (now or datetime.now(tz)).astimezone(tz).date()


def _window_for_day(day: date, tz: ZoneInfo) -> DateWindow:
    start_local = datetime.combine(day, time.min, tzinfo=tz)
    end_local = datetime.combine(day, time.max, tzinfo=tz)
    return DateWindow(
        label=day.isoformat(),
        from_iso=_newsapi_timestamp(start_local.astimezone(UTC)),
        to_iso=_newsapi_timestamp(end_local.astimezone(UTC)),
    )


def _newsapi_timestamp(value: datetime) -> str:
    return value.replace(microsecond=0, tzinfo=None).isoformat()
