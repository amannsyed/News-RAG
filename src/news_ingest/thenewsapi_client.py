from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class TheNewsApiPage:
    found: int
    returned: int
    limit: int
    page: int
    articles: list[dict[str, Any]]


class TheNewsApiRequestError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"TheNewsAPI error {code}: {message}")
        self.code = code
        self.message = message


class TheNewsApiClient:
    def __init__(self, api_token: str, base_url: str = "https://api.thenewsapi.com/v1") -> None:
        self.api_token = api_token
        self.base_url = base_url.rstrip("/")

    def all_news_page(
        self,
        *,
        query: str,
        language: str | None,
        page_size: int,
        page: int,
        published_on: str | None = None,
    ) -> TheNewsApiPage:
        params: dict[str, Any] = {
            "api_token": self.api_token,
            "search": query,
            "limit": page_size,
            "page": page,
        }
        if language:
            params["language"] = language
        if published_on:
            params["published_on"] = published_on

        response = requests.get(f"{self.base_url}/news/all", params=params, timeout=30)
        try:
            payload = response.json()
        except ValueError as exc:
            raise TheNewsApiRequestError(str(response.status_code), response.text[:500]) from exc

        if response.status_code >= 400 or "error" in payload:
            raise TheNewsApiRequestError(_error_code(payload, response.status_code), _error_message(payload))

        meta = payload.get("meta") or {}
        return TheNewsApiPage(
            found=int(meta.get("found", 0) or 0),
            returned=int(meta.get("returned", len(payload.get("data") or [])) or 0),
            limit=int(meta.get("limit", page_size) or page_size),
            page=int(meta.get("page", page) or page),
            articles=list(payload.get("data") or []),
        )


def _error_code(payload: dict[str, Any], status_code: int) -> str:
    return str(payload.get("code") or payload.get("status") or status_code)


def _error_message(payload: dict[str, Any]) -> str:
    errors = payload.get("errors")
    if isinstance(errors, dict):
        return "; ".join(f"{key}: {value}" for key, value in errors.items())
    if isinstance(errors, list):
        return "; ".join(str(error) for error in errors)
    return str(payload.get("message") or payload.get("error") or "TheNewsAPI request failed")


def thenewsapi_to_article(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": {
            "id": article.get("source"),
            "name": article.get("source") or "TheNewsAPI",
            "provider": "thenewsapi",
            "language": article.get("language"),
            "locale": article.get("locale"),
            "categories": article.get("categories"),
            "uuid": article.get("uuid"),
        },
        "author": None,
        "title": article.get("title"),
        "description": article.get("description"),
        "url": article.get("url"),
        "urlToImage": article.get("image_url"),
        "publishedAt": article.get("published_at"),
        "content": article.get("snippet") or article.get("description"),
        "raw_thenewsapi_article": article,
    }
