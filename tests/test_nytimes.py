from types import SimpleNamespace

import pytest

from news_ingest.ingest import IngestSummary
from news_ingest.nytimes_client import NYTimesClient, NYTimesRequestError, nytimes_to_article
from news_ingest.nytimes_ingest import _ingest_nytimes_window, _is_nytimes_rate_limited, _nyt_date


def test_nytimes_to_article_maps_to_shared_shape() -> None:
    article = nytimes_to_article(
        {
            "uri": "nyt://article/123",
            "web_url": "https://www.nytimes.com/story.html",
            "headline": {"main": "Main Headline"},
            "abstract": "Abstract",
            "lead_paragraph": "Lead paragraph",
            "pub_date": "2026-06-29T10:00:00+0000",
            "source": "The New York Times",
            "byline": {"original": "By Reporter"},
            "multimedia": {"default": {"url": "images/2026/06/29/photo.jpg"}},
        }
    )

    assert article["source"]["id"] == "nyt://article/123"
    assert article["source"]["provider"] == "nytimes"
    assert article["title"] == "Main Headline"
    assert article["content"] == "Lead paragraph"
    assert article["urlToImage"] == "https://www.nytimes.com/images/2026/06/29/photo.jpg"


def test_nyt_date_uses_yyyymmdd() -> None:
    assert _nyt_date("2026-06-29T00:00:00") == "20260629"


def test_nytimes_client_parses_fault_payload(monkeypatch) -> None:
    class Response:
        status_code = 429
        text = '{"fault":{"faultstring":"Rate limit exceeded"}}'

        def json(self):
            return {"fault": {"faultstring": "Rate limit exceeded"}}

    def fake_get(url, params, timeout):
        assert params["api-key"] == "key"
        assert params["page"] == 0
        return Response()

    monkeypatch.setattr("news_ingest.nytimes_client.requests.get", fake_get)

    with pytest.raises(NYTimesRequestError) as exc_info:
        NYTimesClient("key").search_page(
            query="climate",
            begin_date="20260629",
            end_date="20260629",
            page=0,
            sort_by="publishedAt",
        )

    assert exc_info.value.code == "Rate limit exceeded"
    assert exc_info.value.message == "Rate limit exceeded"


def test_nytimes_ingest_stops_on_rate_limit() -> None:
    def raise_quota(**kwargs):
        raise NYTimesRequestError("429", "rate limit")

    summary = IngestSummary()
    _ingest_nytimes_window(
        conn=SimpleNamespace(commit=lambda: None, rollback=lambda: None),
        client=SimpleNamespace(search_page=raise_quota),
        settings=SimpleNamespace(nytimes_page_size=10, max_pages_per_window=1, sort_by="publishedAt"),
        query="climate",
        filter_query=None,
        window=SimpleNamespace(label="2026-06-29", from_iso="2026-06-29T00:00:00", to_iso="2026-06-29T23:59:59"),
        summary=summary,
        max_calls=10,
        sleep_seconds=0,
    )

    assert summary.errors == 0
    assert summary.stopped_reason == "rateLimited"


def test_nytimes_rate_limit_detection_matches_policy_quota_violation() -> None:
    exc = NYTimesRequestError("policies.ratelimit.QuotaViolation", "Rate limit quota violation")

    assert _is_nytimes_rate_limited(exc)
