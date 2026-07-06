from types import SimpleNamespace

import pytest

from news_ingest.ingest import IngestSummary
from news_ingest.thenewsapi_client import TheNewsApiClient, TheNewsApiRequestError, thenewsapi_to_article
from news_ingest.thenewsapi_ingest import _ingest_thenewsapi_window, _is_thenewsapi_rate_limited


def test_thenewsapi_to_article_maps_to_shared_shape() -> None:
    article = thenewsapi_to_article(
        {
            "uuid": "abc",
            "title": "Title",
            "description": "Description",
            "snippet": "Snippet",
            "url": "https://example.com/news",
            "image_url": "https://example.com/image.jpg",
            "published_at": "2026-07-04T10:00:00.000000Z",
            "source": "Example News",
            "language": "en",
            "locale": "us",
            "categories": ["tech"],
        }
    )

    assert article["source"]["id"] == "Example News"
    assert article["source"]["provider"] == "thenewsapi"
    assert article["source"]["uuid"] == "abc"
    assert article["content"] == "Snippet"
    assert article["urlToImage"] == "https://example.com/image.jpg"


def test_thenewsapi_client_uses_all_news_params(monkeypatch) -> None:
    calls = {}

    class Response:
        status_code = 200
        text = '{"meta":{"found":1,"returned":1,"limit":10,"page":1},"data":[{"title":"Title"}]}'

        def json(self):
            return {"meta": {"found": 1, "returned": 1, "limit": 10, "page": 1}, "data": [{"title": "Title"}]}

    def fake_get(url, params, timeout):
        calls["url"] = url
        calls["params"] = params
        calls["timeout"] = timeout
        return Response()

    monkeypatch.setattr("news_ingest.thenewsapi_client.requests.get", fake_get)

    page = TheNewsApiClient("token").all_news_page(
        query="technology",
        language="en",
        page_size=10,
        page=1,
        published_on="2026-07-04",
    )

    assert calls["url"] == "https://api.thenewsapi.com/v1/news/all"
    assert calls["params"]["api_token"] == "token"
    assert calls["params"]["search"] == "technology"
    assert calls["params"]["published_on"] == "2026-07-04"
    assert page.found == 1
    assert page.articles == [{"title": "Title"}]


def test_thenewsapi_client_parses_error_payload(monkeypatch) -> None:
    class Response:
        status_code = 429
        text = '{"errors":{"limit":"Rate limit exceeded"}}'

        def json(self):
            return {"errors": {"limit": "Rate limit exceeded"}}

    def fake_get(url, params, timeout):
        return Response()

    monkeypatch.setattr("news_ingest.thenewsapi_client.requests.get", fake_get)

    with pytest.raises(TheNewsApiRequestError) as exc_info:
        TheNewsApiClient("token").all_news_page(query="technology", language="en", page_size=10, page=1)

    assert exc_info.value.code == "429"
    assert exc_info.value.message == "limit: Rate limit exceeded"


def test_thenewsapi_rate_limit_detection() -> None:
    assert _is_thenewsapi_rate_limited(TheNewsApiRequestError("429", "Rate limit exceeded"))


def test_thenewsapi_ingest_stops_on_quota_limit() -> None:
    def raise_quota(**kwargs):
        raise TheNewsApiRequestError("429", "Rate limit exceeded")

    summary = IngestSummary()
    _ingest_thenewsapi_window(
        conn=SimpleNamespace(commit=lambda: None, rollback=lambda: None),
        client=SimpleNamespace(all_news_page=raise_quota),
        settings=SimpleNamespace(thenewsapi_page_size=10, max_pages_per_window=1, language="en"),
        query="technology",
        window=SimpleNamespace(label="2026-07-04"),
        summary=summary,
        max_calls=10,
        sleep_seconds=0,
    )

    assert summary.errors == 0
    assert summary.stopped_reason == "rateLimited"


def test_thenewsapi_ingest_uses_published_on_window() -> None:
    calls = {}

    def all_news_page(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(articles=[], found=0, returned=0)

    summary = IngestSummary()
    _ingest_thenewsapi_window(
        conn=SimpleNamespace(commit=lambda: None, rollback=lambda: None),
        client=SimpleNamespace(all_news_page=all_news_page),
        settings=SimpleNamespace(thenewsapi_page_size=10, max_pages_per_window=1, language="en"),
        query="technology",
        window=SimpleNamespace(label="2026-07-04"),
        summary=summary,
        max_calls=10,
        sleep_seconds=0,
    )

    assert calls["query"] == "technology"
    assert calls["published_on"] == "2026-07-04"
    assert calls["page"] == 1
    assert summary.calls_used == 1
