from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException


@dataclass(frozen=True)
class NewsApiPage:
    status: str
    total_results: int
    articles: list[dict[str, Any]]
    page: int


class NewsApiRequestError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"NewsAPI error {code}: {message}")
        self.code = code
        self.message = message


class NewsApiEverythingClient:
    def __init__(self, api_key: str) -> None:
        self._client = NewsApiClient(api_key=api_key)

    def get_everything_page(
        self,
        *,
        query: str,
        from_iso: str,
        to_iso: str,
        page_size: int,
        page: int,
        language: str | None,
        sort_by: str,
    ) -> NewsApiPage:
        try:
            response = self._client.get_everything(
                q=query,
                from_param=from_iso,
                to=to_iso,
                language=language,
                sort_by=sort_by,
                page_size=page_size,
                page=page,
            )
        except NewsAPIException as exc:
            payload = exc.args[0] if exc.args and isinstance(exc.args[0], dict) else {}
            code = str(payload.get("code", "unknown"))
            message = str(payload.get("message", exc))
            raise NewsApiRequestError(code, message) from exc

        if response.get("status") != "ok":
            code = response.get("code", "unknown")
            message = response.get("message", "NewsAPI request failed")
            raise NewsApiRequestError(str(code), str(message))

        return NewsApiPage(
            status=response["status"],
            total_results=int(response.get("totalResults", 0)),
            articles=list(response.get("articles") or []),
            page=page,
        )
