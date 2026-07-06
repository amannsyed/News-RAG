from types import SimpleNamespace

import pytest

from news_ingest.currents_client import CurrentsClient, CurrentsRequestError, currents_to_article
from news_ingest.currents_ingest import _currents_timestamp, _ingest_currents_window
from news_ingest.ingest import IngestSummary


def test_currents_to_article_maps_to_shared_shape() -> None:
    article = currents_to_article(
        {
            "id": "abc",
            "title": "Title",
            "description": "Description",
            "url": "https://example.com/news",
            "author": "Author",
            "image": "https://example.com/image.jpg",
            "language": "en",
            "category": ["technology"],
            "published": "2026-06-29 10:00:00 +0000",
        }
    )

    assert article["source"]["id"] == "abc"
    assert article["source"]["provider"] == "currents"
    assert article["content"] == "Description"
    assert article["urlToImage"] == "https://example.com/image.jpg"
    assert article["publishedAt"] == "2026-06-29 10:00:00 +0000"


def test_currents_client_uses_msg_for_error_message(monkeypatch) -> None:
    class Response:
        status_code = 429
        text = '{"status":"429","msg":"Daily quota exceeded."}'

        def json(self):
            return {"status": "429", "msg": "Daily quota exceeded."}

    def fake_get(url, params, timeout):
        assert params["apiKey"] == "key"
        return Response()

    monkeypatch.setattr("news_ingest.currents_client.requests.get", fake_get)

    with pytest.raises(CurrentsRequestError) as exc_info:
        CurrentsClient("key").search_page(
            query="technology",
            from_iso="2026-06-29",
            to_iso="2026-06-29",
            page_size=5,
            page=1,
            language="en",
        )

    assert exc_info.value.code == "429"
    assert exc_info.value.message == "Daily quota exceeded."


def test_currents_ingest_stops_on_bad_request() -> None:
    def raise_bad_request(**kwargs):
        raise CurrentsRequestError("400", "Bad date format")

    summary = IngestSummary()
    _ingest_currents_window(
        conn=SimpleNamespace(commit=lambda: None, rollback=lambda: None),
        client=SimpleNamespace(search_page=raise_bad_request),
        settings=SimpleNamespace(currents_page_size=20, max_pages_per_window=1, language="en"),
        query="technology",
        window=SimpleNamespace(label="2026-06-29", from_iso="2026-06-29T00:00:00", to_iso="2026-06-29T23:59:59"),
        summary=summary,
        max_calls=10,
        sleep_seconds=0,
    )

    assert summary.errors == 1
    assert summary.stopped_reason == "currents_400"


def test_currents_timestamp_appends_z() -> None:
    assert _currents_timestamp("2026-06-29T00:00:00") == "2026-06-29T00:00:00Z"
    assert _currents_timestamp("2026-06-29T00:00:00Z") == "2026-06-29T00:00:00Z"


def test_currents_client_uses_nested_errors_for_error_message(monkeypatch) -> None:
    class Response:
        status_code = 400
        text = '{"status":"400","details":{"errors":{"page_size":"Max page_size for Free tier is 20"}}}'

        def json(self):
            return {"status": "400", "details": {"errors": {"page_size": "Max page_size for Free tier is 20"}}}

    def fake_get(url, params, timeout):
        return Response()

    monkeypatch.setattr("news_ingest.currents_client.requests.get", fake_get)

    with pytest.raises(CurrentsRequestError) as exc_info:
        CurrentsClient("key").search_page(
            query="technology",
            from_iso="2026-07-04T00:00:00Z",
            to_iso="2026-07-04T23:59:59Z",
            page_size=100,
            page=1,
            language="en",
        )

    assert exc_info.value.message == "page_size: Max page_size for Free tier is 20"
