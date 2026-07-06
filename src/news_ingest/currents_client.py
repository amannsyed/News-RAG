from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class CurrentsPage:
    status: str
    articles: list[dict[str, Any]]
    page: int


class CurrentsRequestError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"Currents API error {code}: {message}")
        self.code = code
        self.message = message


class CurrentsClient:
    def __init__(self, api_key: str, base_url: str = "https://api.currentsapi.services/v1") -> None:
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
    ) -> CurrentsPage:
        params: dict[str, Any] = {
            "keywords": query,
            "start_date": from_iso,
            "end_date": to_iso,
            "page_size": page_size,
            "page_number": page,
            "apiKey": self.api_key,
        }
        if language:
            params["language"] = language

        response = requests.get(
            f"{self.base_url}/search",
            params=params,
            timeout=30,
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise CurrentsRequestError(str(response.status_code), response.text[:500]) from exc

        status = str(payload.get("status", response.status_code))
        if response.status_code >= 400 or status not in ("ok", "success", "200", "201", "None"):
            code = str(payload.get("code", status))
            message = _error_message(payload)
            raise CurrentsRequestError(code, message)

        return CurrentsPage(
            status=str(payload.get("status", "ok")),
            articles=list(payload.get("news") or []),
            page=page,
        )


def _error_message(payload: dict[str, Any]) -> str:
    details = payload.get("details")
    if isinstance(details, dict):
        errors = details.get("errors")
        if isinstance(errors, dict):
            return "; ".join(f"{field}: {message}" for field, message in errors.items())
        if details.get("message"):
            return str(details["message"])
    return str(
        payload.get("message")
        or payload.get("msg")
        or payload.get("description")
        or payload.get("error")
        or "Currents request failed"
    )


def currents_to_article(article: dict[str, Any]) -> dict[str, Any]:
    published = article.get("published") or article.get("publishedAt")
    return {
        "source": {
            "id": article.get("id"),
            "name": article.get("source") or article.get("source_name") or "Currents",
            "provider": "currents",
            "category": article.get("category"),
            "language": article.get("language"),
        },
        "author": article.get("author"),
        "title": article.get("title"),
        "description": article.get("description"),
        "url": article.get("url"),
        "urlToImage": article.get("image"),
        "publishedAt": published,
        "content": article.get("description"),
        "raw_currents_article": article,
    }
