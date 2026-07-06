from types import SimpleNamespace

import pytest

from news_ingest.gnews_client import GNewsClient, GNewsRequestError, gnews_to_article
from news_ingest.gnews_ingest import _gnews_timestamp, _ingest_gnews_window
from news_ingest.ingest import IngestSummary


def test_gnews_to_article_maps_to_shared_shape() -> None:
    article = gnews_to_article(
        {
            "title": "Title",
            "description": "Description",
            "content": "Content",
            "url": "https://example.com/news",
            "image": "https://example.com/image.jpg",
            "publishedAt": "2026-06-29T10:00:00Z",
            "source": {"name": "Example", "url": "https://example.com"},
        }
    )

    assert article["source"]["name"] == "Example"
    assert article["source"]["provider"] == "gnews"
    assert article["content"] == "Content"
    assert article["urlToImage"] == "https://example.com/image.jpg"


def test_gnews_timestamp_appends_z() -> None:
    assert _gnews_timestamp("2026-06-29T10:00:00") == "2026-06-29T10:00:00Z"
    assert _gnews_timestamp("2026-06-29T10:00:00Z") == "2026-06-29T10:00:00Z"


def test_gnews_client_uses_errors_for_error_message(monkeypatch) -> None:
    class Response:
        status_code = 403
        text = '{"errors":["Daily request limit reached"]}'

        def json(self):
            return {"errors": ["Daily request limit reached"]}

    def fake_get(url, params, timeout):
        assert params["apikey"] == "key"
        return Response()

    monkeypatch.setattr("news_ingest.gnews_client.requests.get", fake_get)

    with pytest.raises(GNewsRequestError) as exc_info:
        GNewsClient("key").search_page(
            query="technology",
            from_iso="2026-06-29T00:00:00Z",
            to_iso="2026-06-29T23:59:59Z",
            page_size=10,
            page=1,
            language="en",
            sort_by="publishedAt",
        )

    assert exc_info.value.code == "403"
    assert exc_info.value.message == "Daily request limit reached"


def test_gnews_ingest_stops_on_quota_limit() -> None:
    def raise_quota(**kwargs):
        raise GNewsRequestError("403", "Daily request limit reached")

    summary = IngestSummary()
    _ingest_gnews_window(
        conn=SimpleNamespace(commit=lambda: None, rollback=lambda: None),
        client=SimpleNamespace(search_page=raise_quota),
        settings=SimpleNamespace(gnews_page_size=10, max_pages_per_window=1, language="en", sort_by="publishedAt"),
        query="technology",
        window=SimpleNamespace(label="2026-06-29", from_iso="2026-06-29T00:00:00", to_iso="2026-06-29T23:59:59"),
        summary=summary,
        max_calls=10,
        sleep_seconds=0,
    )

    assert summary.errors == 0
    assert summary.stopped_reason == "rateLimited"
