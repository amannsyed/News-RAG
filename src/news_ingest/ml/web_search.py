from __future__ import annotations

import hashlib
import html
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from xml.etree import ElementTree
from urllib.request import Request, urlopen

from psycopg.types.json import Jsonb

from news_ingest.db import normalize_url, url_hash
from news_ingest.ml.schema import ensure_ml_schema


DEFAULT_WEB_SEARCH_LIMIT = 5
DEFAULT_WEB_SEARCH_TTL_SECONDS = 21600


@dataclass(frozen=True)
class WebSearchResult:
    id: int | None
    title: str
    url: str
    snippet: str | None
    provider: str = "duckduckgo"
    published_at: str | None = None
    fetched_at: str | None = None
    score: float | None = None


def fetch_and_index_web_search(conn, *, query: str, limit: int = DEFAULT_WEB_SEARCH_LIMIT, ttl_seconds: int | None = None) -> list[dict[str, Any]]:
    ensure_ml_schema(conn)
    ttl_seconds = ttl_seconds if ttl_seconds is not None else int(os.getenv("WEB_SEARCH_TTL_SECONDS", str(DEFAULT_WEB_SEARCH_TTL_SECONDS)))
    cached = _cached_results(conn, query=query, limit=limit, ttl_seconds=ttl_seconds)
    if len(cached) >= limit:
        return [result_to_citation(item, rank=index + 1) for index, item in enumerate(cached[:limit])]

    provider = os.getenv("WEB_SEARCH_PROVIDER", "bing_news").strip().lower() or "bing_news"
    if provider == "duckduckgo":
        results = duckduckgo_search(query=query, limit=limit)
    elif provider == "bing_news":
        results = bing_news_search(query=query, limit=limit)
    else:
        raise ValueError("WEB_SEARCH_PROVIDER supports 'bing_news' or 'duckduckgo'")

    if not results and provider != "bing_news":
        results = bing_news_search(query=query, limit=limit)
    indexed = [_upsert_result(conn, query=query, result=result, rank=index + 1) for index, result in enumerate(results)]
    conn.commit()
    return [result_to_citation(item, rank=index + 1) for index, item in enumerate(indexed[:limit])]


def result_to_citation(result: WebSearchResult, *, rank: int) -> dict[str, Any]:
    return {
        "source_type": "web_search",
        "citation_marker": "",
        "web_search_id": result.id,
        "article_id": None,
        "title": result.title,
        "url": result.url,
        "provider": f"web:{result.provider}",
        "published_at": result.published_at,
        "rrf_score": result.score,
        "stream_ranks": {"web_search": rank},
        "snippet": result.snippet,
        "char_start": None,
        "char_end": None,
    }


def duckduckgo_search(*, query: str, limit: int) -> list[WebSearchResult]:
    url = "https://duckduckgo.com/html/?q=" + quote_plus(query)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 NewsRAG/1.0"})
    with urlopen(request, timeout=int(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "15"))) as response:
        body = response.read().decode("utf-8", errors="replace")
    parser = DuckDuckGoHTMLParser()
    parser.feed(body)
    deduped: list[WebSearchResult] = []
    seen: set[str] = set()
    for item in parser.results:
        normalized = normalize_url(item.url) or item.url
        key = url_hash(normalized)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def bing_news_search(*, query: str, limit: int) -> list[WebSearchResult]:
    url = "https://www.bing.com/news/search?q=" + quote_plus(query) + "&format=rss"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 NewsRAG/1.0"})
    with urlopen(request, timeout=int(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "15"))) as response:
        body = response.read()
    root = ElementTree.fromstring(body)
    results: list[WebSearchResult] = []
    for item in root.findall("./channel/item"):
        title = _clean_text(item.findtext("title") or "")
        link = _clean_bing_url(_clean_text(item.findtext("link") or ""))
        description = _clean_text(re.sub(r"<[^>]+>", " ", item.findtext("description") or ""))
        published_at = _clean_text(item.findtext("pubDate") or "") or None
        if not title or not link:
            continue
        results.append(WebSearchResult(id=None, title=title, url=link, snippet=description or None, provider="bing_news", published_at=published_at))
        if len(results) >= limit:
            break
    return results


class DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[WebSearchResult] = []
        self._in_title = False
        self._in_snippet = False
        self._title_chunks: list[str] = []
        self._snippet_chunks: list[str] = []
        self._current_url: str | None = None
        self._pending_result_index: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {key: value or "" for key, value in attrs}
        class_name = attrs_dict.get("class", "")
        if tag == "a" and "result__a" in class_name:
            self._in_title = True
            self._title_chunks = []
            self._current_url = _clean_duckduckgo_url(attrs_dict.get("href", ""))
        elif tag in {"a", "div"} and "result__snippet" in class_name:
            self._in_snippet = True
            self._snippet_chunks = []

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_chunks.append(data)
        elif self._in_snippet:
            self._snippet_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            title = _clean_text("".join(self._title_chunks))
            if title and self._current_url:
                self.results.append(WebSearchResult(id=None, title=title, url=self._current_url, snippet=None))
                self._pending_result_index = len(self.results) - 1
            self._in_title = False
            self._title_chunks = []
            self._current_url = None
        elif self._in_snippet and tag in {"a", "div"}:
            snippet = _clean_text("".join(self._snippet_chunks))
            if snippet and self._pending_result_index is not None:
                item = self.results[self._pending_result_index]
                self.results[self._pending_result_index] = WebSearchResult(id=item.id, title=item.title, url=item.url, snippet=snippet, provider=item.provider)
                self._pending_result_index = None
            self._in_snippet = False
            self._snippet_chunks = []


def _clean_bing_url(value: str) -> str:
    parsed = urlparse(value)
    target = parse_qs(parsed.query).get("url", [""])[0]
    return unquote(target) if target else value


def _clean_duckduckgo_url(value: str) -> str:
    value = html.unescape(value)
    parsed = urlparse(value)
    if parsed.path == "/l/":
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return value


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def _cached_results(conn, *, query: str, limit: int, ttl_seconds: int) -> list[WebSearchResult]:
    cutoff = datetime.now(UTC) - timedelta(seconds=ttl_seconds)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, title, url, snippet, provider, published_at::text AS published_at, fetched_at::text AS fetched_at, score
            FROM web_search_documents
            WHERE query_hash = %(query_hash)s
              AND fetched_at >= %(cutoff)s
            ORDER BY score DESC NULLS LAST, fetched_at DESC
            LIMIT %(limit)s;
            """,
            {"query_hash": _query_hash(query), "cutoff": cutoff, "limit": limit},
        )
        return [_row_to_result(row) for row in cur.fetchall()]


def _upsert_result(conn, *, query: str, result: WebSearchResult, rank: int) -> WebSearchResult:
    normalized_url = normalize_url(result.url) or result.url
    score = 1.0 / rank
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO web_search_documents(query, query_hash, provider, title, url, url_hash, snippet, document_text, raw_payload, score)
            VALUES (%(query)s, %(query_hash)s, %(provider)s, %(title)s, %(url)s, %(url_hash)s, %(snippet)s, %(document_text)s, %(raw_payload)s, %(score)s)
            ON CONFLICT (url_hash, query_hash) DO UPDATE SET
                title = COALESCE(EXCLUDED.title, web_search_documents.title),
                snippet = COALESCE(EXCLUDED.snippet, web_search_documents.snippet),
                document_text = COALESCE(EXCLUDED.document_text, web_search_documents.document_text),
                raw_payload = EXCLUDED.raw_payload,
                score = EXCLUDED.score,
                fetched_at = now()
            RETURNING id, title, url, snippet, provider, published_at::text AS published_at, fetched_at::text AS fetched_at, score;
            """,
            {
                "query": query,
                "query_hash": _query_hash(query),
                "provider": result.provider,
                "title": result.title,
                "url": normalized_url,
                "url_hash": url_hash(normalized_url),
                "snippet": result.snippet,
                "document_text": build_web_document(result),
                "raw_payload": Jsonb({"url": result.url, "title": result.title, "snippet": result.snippet, "provider": result.provider}),
                "score": score,
            },
        )
        return _row_to_result(cur.fetchone())


def build_web_document(result: WebSearchResult) -> str:
    return _clean_text("\n".join(part for part in [result.title, result.snippet or "", result.url] if part))


def _row_to_result(row: dict[str, Any]) -> WebSearchResult:
    return WebSearchResult(
        id=int(row["id"]),
        title=row["title"],
        url=row["url"],
        snippet=row.get("snippet"),
        provider=row.get("provider") or "duckduckgo",
        published_at=row.get("published_at"),
        fetched_at=row.get("fetched_at"),
        score=float(row["score"]) if row.get("score") is not None else None,
    )


def _query_hash(query: str) -> str:
    return hashlib.sha256(" ".join(query.lower().split()).encode("utf-8")).hexdigest()
