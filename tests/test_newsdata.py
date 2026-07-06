from types import SimpleNamespace

import pytest

from news_ingest.ingest import IngestSummary
from news_ingest.newsdata_client import NewsDataClient, NewsDataRequestError, newsdata_to_article
from news_ingest.newsdata_ingest import _ingest_newsdata_window, _is_newsdata_rate_limited


def test_newsdata_to_article_maps_to_shared_shape() -> None:
    article = newsdata_to_article(
        {
            "article_id": "abc",
            "title": "Title",
            "description": "Description",
            "content": "Content",
            "link": "https://example.com/news",
            "image_url": "https://example.com/image.jpg",
            "creator": ["Author One", "Author Two"],
            "pubDate": "2026-07-04 10:00:00",
            "source_id": "example",
            "source_name": "Example News",
            "language": "english",
            "category": ["technology"],
        }
    )

    assert article["source"]["id"] == "example"
    assert article["source"]["provider"] == "newsdata"
    assert article["author"] == "Author One, Author Two"
    assert article["content"] == "Content"
    assert article["url"] == "https://example.com/news"


def test_newsdata_client_parses_results_error_message(monkeypatch) -> None:
    class Response:
        status_code = 429
        text = '{"status":"error","results":{"message":"Rate limit exceeded"}}'

        def json(self):
            return {"status": "error", "results": {"message": "Rate limit exceeded"}}

    def fake_get(url, params, timeout):
        assert params["apikey"] == "key"
        return Response()

    monkeypatch.setattr("news_ingest.newsdata_client.requests.get", fake_get)

    with pytest.raises(NewsDataRequestError) as exc_info:
        NewsDataClient("key").archive_page(
            query="technology",
            language="en",
            page_size=10,
            from_date="2026-07-04",
            to_date="2026-07-04",
        )

    assert exc_info.value.message == "Rate limit exceeded"


def test_newsdata_rate_limit_detection() -> None:
    assert _is_newsdata_rate_limited(NewsDataRequestError("429", "Rate limit exceeded"))


def test_newsdata_ingest_stops_on_quota_limit() -> None:
    def raise_quota(**kwargs):
        raise NewsDataRequestError("429", "Rate limit exceeded")

    summary = IngestSummary()
    _ingest_newsdata_window(
        conn=SimpleNamespace(commit=lambda: None, rollback=lambda: None),
        client=SimpleNamespace(archive_page=raise_quota),
        settings=SimpleNamespace(newsdata_page_size=10, max_pages_per_window=1, language="en"),
        query="technology",
        window=SimpleNamespace(label="2026-07-04"),
        summary=summary,
        max_calls=10,
        sleep_seconds=0,
    )

    assert summary.errors == 0
    assert summary.stopped_reason == "rateLimited"


def test_newsdata_ingest_uses_archive_endpoint() -> None:
    calls = {}

    def archive_page(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(articles=[], total_results=0, next_page=None)

    summary = IngestSummary()
    _ingest_newsdata_window(
        conn=SimpleNamespace(commit=lambda: None, rollback=lambda: None),
        client=SimpleNamespace(archive_page=archive_page),
        settings=SimpleNamespace(newsdata_page_size=10, max_pages_per_window=1, language="en"),
        query="technology",
        window=SimpleNamespace(label="2026-07-04"),
        summary=summary,
        max_calls=10,
        sleep_seconds=0,
    )

    assert calls["from_date"] == "2026-07-04"
    assert calls["to_date"] == "2026-07-04"
    assert summary.calls_used == 1


def test_newsdata_ingest_latest_only_uses_latest_endpoint() -> None:
    calls = {}

    def latest_page(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(articles=[], total_results=0, next_page=None)

    summary = IngestSummary()
    _ingest_newsdata_window(
        conn=SimpleNamespace(commit=lambda: None, rollback=lambda: None),
        client=SimpleNamespace(latest_page=latest_page),
        settings=SimpleNamespace(newsdata_page_size=10, max_pages_per_window=1, language="en"),
        query="technology",
        window=None,
        summary=summary,
        max_calls=10,
        sleep_seconds=0,
        latest_only=True,
    )

    assert calls["query"] == "technology"
    assert summary.calls_used == 1
