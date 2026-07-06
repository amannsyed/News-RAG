from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class NewsDataPage:
    total_results: int | None
    articles: list[dict[str, Any]]
    next_page: str | None
    page: str | None


class NewsDataRequestError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"NewsData.io API error {code}: {message}")
        self.code = code
        self.message = message


class NewsDataClient:
    def __init__(self, api_key: str, base_url: str = "https://newsdata.io/api/1") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def latest_page(
        self,
        *,
        query: str,
        language: str | None,
        page_size: int,
        page: str | None = None,
    ) -> NewsDataPage:
        params: dict[str, Any] = {
            "apikey": self.api_key,
            "q": query,
            "size": page_size,
        }
        if language:
            params["language"] = language
        if page:
            params["page"] = page

        return self._request("latest", params=params, page=page)

    def archive_page(
        self,
        *,
        query: str,
        language: str | None,
        page_size: int,
        from_date: str,
        to_date: str,
        page: str | None = None,
    ) -> NewsDataPage:
        params: dict[str, Any] = {
            "apikey": self.api_key,
            "q": query,
            "size": page_size,
            "from_date": from_date,
            "to_date": to_date,
        }
        if language:
            params["language"] = language
        if page:
            params["page"] = page

        return self._request("archive", params=params, page=page)

    def _request(self, endpoint: str, *, params: dict[str, Any], page: str | None) -> NewsDataPage:
        response = requests.get(f"{self.base_url}/{endpoint}", params=params, timeout=30)
        try:
            payload = response.json()
        except ValueError as exc:
            raise NewsDataRequestError(str(response.status_code), response.text[:500]) from exc

        status = str(payload.get("status", response.status_code)).lower()
        if response.status_code >= 400 or status not in {"success", "200", "ok"}:
            raise NewsDataRequestError(_error_code(payload, response.status_code), _error_message(payload))

        return NewsDataPage(
            total_results=_optional_int(payload.get("totalResults")),
            articles=list(payload.get("results") or []),
            next_page=payload.get("nextPage"),
            page=page,
        )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _error_code(payload: dict[str, Any], status_code: int) -> str:
    return str(payload.get("code") or payload.get("status") or status_code)


def _error_message(payload: dict[str, Any]) -> str:
    results = payload.get("results")
    if isinstance(results, dict):
        message = results.get("message") or results.get("error")
        if message:
            return str(message)
    return str(
        payload.get("message")
        or payload.get("msg")
        or payload.get("error")
        or payload.get("detail")
        or "NewsData.io request failed"
    )


def newsdata_to_article(article: dict[str, Any]) -> dict[str, Any]:
    source_name = article.get("source_name") or article.get("source_id") or "NewsData.io"
    creators = article.get("creator") or []
    author = ", ".join(str(value) for value in creators) if isinstance(creators, list) else creators
    return {
        "source": {
            "id": article.get("source_id"),
            "name": source_name,
            "url": article.get("source_url"),
            "icon": article.get("source_icon"),
            "provider": "newsdata",
            "language": article.get("language"),
            "country": article.get("country"),
            "category": article.get("category"),
            "article_id": article.get("article_id"),
        },
        "author": author,
        "title": article.get("title"),
        "description": article.get("description"),
        "url": article.get("link"),
        "urlToImage": article.get("image_url"),
        "publishedAt": article.get("pubDate"),
        "content": article.get("content") or article.get("description"),
        "raw_newsdata_article": article,
    }
