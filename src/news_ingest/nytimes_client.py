from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class NYTimesPage:
    hits: int
    offset: int
    articles: list[dict[str, Any]]
    page: int


class NYTimesRequestError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"NYTimes API error {code}: {message}")
        self.code = code
        self.message = message


class NYTimesClient:
    def __init__(self, api_key: str, base_url: str = "https://api.nytimes.com/svc/search/v2") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def search_page(
        self,
        *,
        query: str,
        begin_date: str,
        end_date: str,
        page: int,
        sort_by: str,
        filter_query: str | None = None,
    ) -> NYTimesPage:
        params: dict[str, Any] = {
            "q": query,
            "begin_date": begin_date,
            "end_date": end_date,
            "page": page,
            "api-key": self.api_key,
        }
        if sort_by == "publishedAt":
            params["sort"] = "newest"
        if filter_query:
            params["fq"] = filter_query

        response = requests.get(f"{self.base_url}/articlesearch.json", params=params, timeout=30)
        try:
            payload = response.json()
        except ValueError as exc:
            raise NYTimesRequestError(str(response.status_code), response.text[:500]) from exc

        if response.status_code >= 400 or payload.get("status") not in ("OK", "ok"):
            fault = payload.get("fault") if isinstance(payload.get("fault"), dict) else {}
            detail = fault.get("detail") if isinstance(fault.get("detail"), dict) else {}
            code = str(payload.get("code") or detail.get("errorcode") or fault.get("faultstring") or response.status_code)
            message = str(payload.get("message") or fault.get("faultstring") or detail.get("errorcode") or "NYTimes request failed")
            raise NYTimesRequestError(code, message)

        response_node = payload.get("response") or {}
        meta = response_node.get("meta") or {}
        return NYTimesPage(
            hits=int(meta.get("hits", 0)),
            offset=int(meta.get("offset", page * 10)),
            articles=list(response_node.get("docs") or []),
            page=page,
        )


def nytimes_to_article(article: dict[str, Any]) -> dict[str, Any]:
    headline = article.get("headline") or {}
    byline = article.get("byline") or {}
    multimedia = article.get("multimedia") or {}
    image_url = _image_url(multimedia)
    source = article.get("source") or "The New York Times"
    return {
        "source": {
            "id": article.get("uri") or article.get("_id"),
            "name": source,
            "provider": "nytimes",
            "section": article.get("section_name") or article.get("sectionName"),
            "subsection": article.get("subsection_name") or article.get("subsectionName"),
            "document_type": article.get("document_type"),
        },
        "author": byline.get("original"),
        "title": headline.get("main") or headline.get("default") or article.get("print_headline"),
        "description": article.get("abstract") or article.get("snippet") or article.get("lead_paragraph"),
        "url": article.get("web_url") or article.get("url"),
        "urlToImage": image_url,
        "publishedAt": article.get("pub_date") or article.get("firstPublished"),
        "content": article.get("lead_paragraph") or article.get("abstract") or article.get("snippet"),
        "raw_nytimes_article": article,
    }


def _image_url(multimedia: Any) -> str | None:
    if isinstance(multimedia, dict):
        for key in ("default", "thumbnail"):
            item = multimedia.get(key)
            if isinstance(item, dict) and item.get("url"):
                return _nyt_image_url(str(item["url"]))
    if isinstance(multimedia, list):
        for item in multimedia:
            if isinstance(item, dict) and item.get("url"):
                return _nyt_image_url(str(item["url"]))
    return None


def _nyt_image_url(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return f"https://www.nytimes.com/{url.lstrip('/')}"
