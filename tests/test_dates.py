from datetime import date, datetime
from zoneinfo import ZoneInfo

from news_ingest.dates import date_range_windows, last_n_days_windows, today_yesterday_windows


def test_today_yesterday_windows_use_configured_timezone() -> None:
    windows = today_yesterday_windows(
        now=datetime(2026, 6, 29, 12, 0, tzinfo=ZoneInfo("Europe/London")),
        tz=ZoneInfo("Europe/London"),
    )

    assert [window.label for window in windows] == ["2026-06-28", "2026-06-29"]
    assert windows[0].from_iso == "2026-06-27T23:00:00"
    assert windows[1].from_iso == "2026-06-28T23:00:00"


def test_last_n_days_windows_include_today() -> None:
    windows = last_n_days_windows(
        3,
        now=datetime(2026, 6, 29, 12, 0, tzinfo=ZoneInfo("Europe/London")),
        tz=ZoneInfo("Europe/London"),
    )

    assert [window.label for window in windows] == ["2026-06-27", "2026-06-28", "2026-06-29"]


def test_date_range_windows_are_inclusive() -> None:
    windows = date_range_windows(date(2026, 6, 25), date(2026, 6, 27), tz=ZoneInfo("Europe/London"))

    assert [window.label for window in windows] == ["2026-06-25", "2026-06-26", "2026-06-27"]
