from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class GNewsPage:
    total_articles: int
    articles: list[dict[str, Any]]
    page: int


class GNewsRequestError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"GNews API error {code}: {message}")
        self.code = code
        self.message = message


class GNewsClient:
    def __init__(self, api_key: str, base_url: str = "https://gnews.io/api/v4") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def search_page(
        self,
        *,
        query: str,
        from_iso: str,
        to_iso: str,
        page_size: int,
        page: int,
        language: str | None,
        sort_by: str,
    ) -> GNewsPage:
        params: dict[str, Any] = {
            "q": query,
            "max": page_size,
            "page": page,
            "from": from_iso,
            "to": to_iso,
            "sortby": "publishedAt" if sort_by == "publishedAt" else "relevance",
            "apikey": self.api_key,
        }
        if language:
            params["lang"] = language

        response = requests.get(f"{self.base_url}/search", params=params, timeout=30)
        try:
            payload = response.json()
        except ValueError as exc:
            raise GNewsRequestError(str(response.status_code), response.text[:500]) from exc

        if response.status_code >= 400:
            raise GNewsRequestError(str(response.status_code), _error_message(payload))

        return GNewsPage(
            total_articles=int(payload.get("totalArticles", 0)),
            articles=list(payload.get("articles") or []),
            page=page,
        )


def _error_message(payload: dict[str, Any]) -> str:
    errors = payload.get("errors")
    if isinstance(errors, list):
        return "; ".join(str(error) for error in errors)
    if isinstance(errors, dict):
        return "; ".join(f"{key}: {value}" for key, value in errors.items())
    return str(payload.get("message") or payload.get("error") or "GNews request failed")


def gnews_to_article(article: dict[str, Any]) -> dict[str, Any]:
    source = article.get("source") or {}
    return {
        "source": {
            "id": source.get("id"),
            "name": source.get("name") or "GNews",
            "url": source.get("url"),
            "country": source.get("country"),
            "provider": "gnews",
            "language": article.get("lang"),
        },
        "author": None,
        "title": article.get("title"),
        "description": article.get("description"),
        "url": article.get("url"),
        "urlToImage": article.get("image"),
        "publishedAt": article.get("publishedAt"),
        "content": article.get("content") or article.get("description"),
        "raw_gnews_article": article,
    }
