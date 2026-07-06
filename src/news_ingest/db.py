from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS article_metadata (
    id BIGSERIAL PRIMARY KEY,
    url TEXT NOT NULL,
    url_hash CHAR(64) NOT NULL UNIQUE,
    provider TEXT NOT NULL DEFAULT 'newsapi',
    source_id TEXT,
    source_name TEXT,
    author TEXT,
    title TEXT,
    description TEXT,
    url_to_image TEXT,
    published_at TIMESTAMPTZ,
    query TEXT NOT NULL,
    window_label TEXT NOT NULL,
    api_page INTEGER NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_raw JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_article_metadata_published_at
    ON article_metadata (published_at DESC);

CREATE INDEX IF NOT EXISTS idx_article_metadata_source_id
    ON article_metadata (source_id);

CREATE INDEX IF NOT EXISTS idx_article_metadata_query_window
    ON article_metadata (query, window_label);

ALTER TABLE article_metadata
    ADD COLUMN IF NOT EXISTS provider TEXT NOT NULL DEFAULT 'newsapi';

CREATE INDEX IF NOT EXISTS idx_article_metadata_provider_query_window
    ON article_metadata (provider, query, window_label);

CREATE TABLE IF NOT EXISTS article_contents (
    article_id BIGINT PRIMARY KEY REFERENCES article_metadata(id) ON DELETE CASCADE,
    content TEXT,
    content_hash CHAR(64),
    raw_article JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE article_contents
    ADD COLUMN IF NOT EXISTS content_hash CHAR(64);

CREATE UNIQUE INDEX IF NOT EXISTS idx_article_contents_content_hash_unique
    ON article_contents (content_hash)
    WHERE content_hash IS NOT NULL;
"""


FIND_CONTENT_DUPLICATE_SQL = """
SELECT article_contents.article_id, article_metadata.url_hash
FROM article_contents
JOIN article_metadata ON article_metadata.id = article_contents.article_id
WHERE article_contents.content_hash = %(content_hash)s
LIMIT 1;
"""


COUNT_QUERY_WINDOW_SQL = """
SELECT COUNT(*) AS article_count
FROM article_metadata
WHERE provider = %(provider)s
  AND query = %(query)s
  AND window_label = %(window_label)s;
"""


UPSERT_METADATA_SQL = """
INSERT INTO article_metadata (
    url,
    url_hash,
    provider,
    source_id,
    source_name,
    author,
    title,
    description,
    url_to_image,
    published_at,
    query,
    window_label,
    api_page,
    fetched_at,
    source_raw
) VALUES (
    %(url)s,
    %(url_hash)s,
    %(provider)s,
    %(source_id)s,
    %(source_name)s,
    %(author)s,
    %(title)s,
    %(description)s,
    %(url_to_image)s,
    %(published_at)s,
    %(query)s,
    %(window_label)s,
    %(api_page)s,
    %(fetched_at)s,
    %(source_raw)s
)
ON CONFLICT (url_hash) DO UPDATE SET
    provider = EXCLUDED.provider,
    source_id = COALESCE(EXCLUDED.source_id, article_metadata.source_id),
    source_name = COALESCE(EXCLUDED.source_name, article_metadata.source_name),
    author = COALESCE(EXCLUDED.author, article_metadata.author),
    title = COALESCE(EXCLUDED.title, article_metadata.title),
    description = COALESCE(EXCLUDED.description, article_metadata.description),
    url_to_image = COALESCE(EXCLUDED.url_to_image, article_metadata.url_to_image),
    published_at = COALESCE(EXCLUDED.published_at, article_metadata.published_at),
    query = EXCLUDED.query,
    window_label = EXCLUDED.window_label,
    api_page = EXCLUDED.api_page,
    fetched_at = EXCLUDED.fetched_at,
    source_raw = EXCLUDED.source_raw,
    updated_at = now()
RETURNING id, (xmax = 0) AS inserted;
"""


UPSERT_CONTENT_SQL = """
INSERT INTO article_contents (
    article_id,
    content,
    content_hash,
    raw_article
) VALUES (
    %(article_id)s,
    %(content)s,
    %(content_hash)s,
    %(raw_article)s
)
ON CONFLICT (article_id) DO UPDATE SET
    content = COALESCE(EXCLUDED.content, article_contents.content),
    content_hash = COALESCE(EXCLUDED.content_hash, article_contents.content_hash),
    raw_article = EXCLUDED.raw_article,
    updated_at = now();
"""


@dataclass(frozen=True)
class UpsertResult:
    inserted: bool
    article_id: int
    duplicate_content: bool = False


def connect(database_url: str) -> psycopg.Connection:
    return psycopg.connect(database_url, row_factory=dict_row)


def ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()


def normalize_url(url: str) -> str:
    split = urlsplit(url.strip())
    scheme = split.scheme.lower() or "https"
    netloc = split.netloc.lower()
    path = split.path or "/"
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(split.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
    ]
    query = urlencode(query_pairs, doseq=True)
    return urlunsplit((scheme, netloc, path, query, ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()


def normalize_content(content: str | None) -> str | None:
    if not content:
        return None
    normalized = re.sub(r"\s+", " ", content).strip()
    return normalized or None


def content_hash(content: str | None) -> str | None:
    normalized = normalize_content(content)
    if normalized is None:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def parse_published_at(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def stored_article_count(conn: psycopg.Connection, *, query: str, window_label: str, provider: str = "newsapi") -> int:
    with conn.cursor() as cur:
        cur.execute(COUNT_QUERY_WINDOW_SQL, {"provider": provider, "query": query, "window_label": window_label})
        row = cur.fetchone()
    return int(row["article_count"] if row is not None else 0)


def upsert_article(
    conn: psycopg.Connection,
    article: dict[str, Any],
    *,
    query: str,
    window_label: str,
    api_page: int,
    fetched_at: datetime,
    provider: str = "newsapi",
) -> UpsertResult | None:
    url = article.get("url")
    if not url:
        return None

    article_url_hash = url_hash(url)
    article_content_hash = content_hash(article.get("content"))

    with conn.cursor() as cur:
        if article_content_hash is not None:
            cur.execute(FIND_CONTENT_DUPLICATE_SQL, {"content_hash": article_content_hash})
            duplicate = cur.fetchone()
            if duplicate is not None and duplicate["url_hash"] != article_url_hash:
                return UpsertResult(
                    inserted=False,
                    article_id=int(duplicate["article_id"]),
                    duplicate_content=True,
                )

        source = article.get("source") or {}
        metadata = {
            "url": url,
            "url_hash": article_url_hash,
            "provider": provider,
            "source_id": source.get("id"),
            "source_name": source.get("name"),
            "author": article.get("author"),
            "title": article.get("title"),
            "description": article.get("description"),
            "url_to_image": article.get("urlToImage"),
            "published_at": parse_published_at(article.get("publishedAt")),
            "query": query,
            "window_label": window_label,
            "api_page": api_page,
            "fetched_at": fetched_at,
            "source_raw": Jsonb(source),
        }

        cur.execute(UPSERT_METADATA_SQL, metadata)
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("metadata upsert did not return a row")
        cur.execute(
            UPSERT_CONTENT_SQL,
            {
                "article_id": row["id"],
                "content": article.get("content"),
                "content_hash": article_content_hash,
                "raw_article": Jsonb(json.loads(json.dumps(article))),
            },
        )

    return UpsertResult(inserted=bool(row["inserted"]), article_id=int(row["id"]))
