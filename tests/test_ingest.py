from types import SimpleNamespace

from news_ingest.ingest import IngestSummary, _ingest_window, build_date_windows, fetch_today_yesterday_articles, parse_query_arg
from news_ingest.newsapi_client import NewsApiPage, NewsApiRequestError


class FakeClient:
    def __init__(self) -> None:
        self.pages = [
            NewsApiPage(status="ok", total_results=150, articles=[{"url": "https://example.com/a"}] * 100, page=1),
            NewsApiPage(status="ok", total_results=150, articles=[{"url": "https://example.com/b"}] * 50, page=2),
        ]

    def get_everything_page(self, **kwargs):
        return self.pages.pop(0)


def test_ingest_window_stops_when_last_page_is_short(monkeypatch) -> None:
    def fake_upsert_article(conn, article, **kwargs):
        return SimpleNamespace(inserted=True)

    monkeypatch.setattr("news_ingest.ingest.upsert_article", fake_upsert_article)

    conn = SimpleNamespace(commit=lambda: None, rollback=lambda: None)
    summary = IngestSummary()

    _ingest_window(
        conn=conn,
        client=FakeClient(),
        settings=SimpleNamespace(page_size=100, max_pages_per_window=1000, language="en", sort_by="publishedAt"),
        query="test",
        window=SimpleNamespace(label="2026-06-29", from_iso="2026-06-29T00:00:00Z", to_iso="2026-06-29T23:59:59Z"),
        summary=summary,
        max_calls=1000,
    )

    assert summary.calls_used == 2
    assert summary.articles_received == 150
    assert summary.inserted == 150


def test_ingest_window_counts_content_duplicates_as_skipped(monkeypatch) -> None:
    def fake_upsert_article(conn, article, **kwargs):
        return SimpleNamespace(inserted=False, duplicate_content=True)

    monkeypatch.setattr("news_ingest.ingest.upsert_article", fake_upsert_article)

    conn = SimpleNamespace(commit=lambda: None, rollback=lambda: None)
    summary = IngestSummary()

    _ingest_window(
        conn=conn,
        client=SimpleNamespace(
            get_everything_page=lambda **kwargs: NewsApiPage(
                status="ok",
                total_results=1,
                articles=[{"url": "https://example.com/duplicate"}],
                page=1,
            )
        ),
        settings=SimpleNamespace(page_size=100, language="en", sort_by="publishedAt"),
        query="test",
        window=SimpleNamespace(label="2026-06-29", from_iso="2026-06-29T00:00:00Z", to_iso="2026-06-29T23:59:59Z"),
        summary=summary,
        max_calls=1000,
    )

    assert summary.skipped == 1
    assert summary.inserted == 0
    assert summary.updated == 0


def test_parse_query_arg_returns_clean_tuple() -> None:
    assert parse_query_arg(" technology ; ; business ;science ") == ("technology", "business", "science")


def test_build_date_windows_rejects_mixed_range_modes() -> None:
    settings = SimpleNamespace(zoneinfo=__import__("zoneinfo").ZoneInfo("Europe/London"))

    try:
        build_date_windows(settings=settings, last_days=7, from_date="2026-06-01", to_date="2026-06-07")
    except ValueError as exc:
        assert "either --last-days or --from-date/--to-date" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_ingest_window_stops_run_on_rate_limit() -> None:
    def raise_rate_limit(**kwargs):
        raise NewsApiRequestError("rateLimited", "too many requests")

    conn = SimpleNamespace(commit=lambda: None, rollback=lambda: None)
    summary = IngestSummary()

    _ingest_window(
        conn=conn,
        client=SimpleNamespace(get_everything_page=raise_rate_limit),
        settings=SimpleNamespace(page_size=100, max_pages_per_window=1, language="en", sort_by="publishedAt"),
        query="test",
        window=SimpleNamespace(label="2026-06-29", from_iso="2026-06-29T00:00:00", to_iso="2026-06-29T23:59:59"),
        summary=summary,
        max_calls=1000,
    )

    assert summary.stopped_reason == "rateLimited"
    assert summary.errors == 0
    assert summary.calls_used == 0


def test_fetch_skips_existing_query_windows_by_default(monkeypatch) -> None:
    calls = {"api": 0}

    class Client:
        def __init__(self, api_key):
            pass

        def get_everything_page(self, **kwargs):
            calls["api"] += 1
            return NewsApiPage(status="ok", total_results=1, articles=[], page=1)

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    settings = SimpleNamespace(
        newsapi_api_key="key",
        database_url="postgresql://example",
        query="test",
        queries=("test",),
        max_calls=10,
        page_size=100,
        max_pages_per_window=1,
        language="en",
        sort_by="publishedAt",
        zoneinfo=__import__("zoneinfo").ZoneInfo("Europe/London"),
    )

    monkeypatch.setattr("news_ingest.ingest.NewsApiEverythingClient", Client)
    monkeypatch.setattr("news_ingest.ingest.connect", lambda database_url: Conn())
    monkeypatch.setattr("news_ingest.ingest.ensure_schema", lambda conn: None)
    monkeypatch.setattr("news_ingest.ingest.today_yesterday_windows", lambda tz: [SimpleNamespace(label="2026-06-29")])
    monkeypatch.setattr("news_ingest.ingest.stored_article_count", lambda conn, query, window_label: 5)

    summary = fetch_today_yesterday_articles(settings=settings)

    assert calls["api"] == 0
    assert summary.skipped_windows == 1
    assert summary.calls_used == 0
