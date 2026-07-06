from __future__ import annotations

import argparse
import logging
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from news_ingest.config import load_settings
from news_ingest.db import connect, ensure_schema
from news_ingest.logging_config import configure_logging
from news_ingest.ml.chunk_client import fetch_token_chunks
from news_ingest.ml.http_client import post_json
from news_ingest.ml.schema import ensure_ml_schema
from news_ingest.ml.document import ensure_document, get_article_document_text


logger = logging.getLogger(__name__)
MODEL_NAME = "google/embeddinggemma-300m"
MODEL_VERSION = "v1"
DOCUMENT_SCOPE = "ml-v1"


@dataclass
class EmbeddingSummary:
    articles_seen: int = 0
    chunks_seen: int = 0
    endpoint_chunks: int = 0
    reused_chunks: int = 0
    inserted_mappings: int = 0
    skipped_empty: int = 0
    errors: int = 0


def vector_literal(values: list[float]) -> str:
    if len(values) != 768:
        raise ValueError(f"expected 768-dimensional embedding, got {len(values)}")
    return "[" + ",".join(f"{float(value):.8g}" for value in values) + "]"


def fetch_embeddings(endpoint_url: str, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    payload = post_json(endpoint_url.rstrip("/") + "/embed", {"texts": texts, "input_type": "document"})
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list):
        raise RuntimeError("embedding endpoint response missing embeddings list")
    return embeddings


def process_embeddings(*, database_url: str, endpoint_url: str, limit: int = 500, batch_size: int = 32, max_tokens: int = 500, overlap_tokens: int = 50) -> EmbeddingSummary:
    summary = EmbeddingSummary()
    with connect(database_url) as conn:
        ensure_schema(conn)
        ensure_ml_schema(conn)
        articles = _load_unembedded_articles(conn, limit=limit)
        summary.articles_seen = len(articles)
        for article in articles:
            try:
                _process_article(conn, article=article, endpoint_url=endpoint_url, batch_size=batch_size, max_tokens=max_tokens, overlap_tokens=overlap_tokens, summary=summary)
            except Exception:
                conn.rollback()
                summary.errors += 1
                logger.exception("Failed embedding processing article_id=%s", article.get("id"))
    return summary


def _load_unembedded_articles(conn, *, limit: int) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT m.id, m.title, m.description, c.content
            FROM article_metadata m
            JOIN article_contents c ON c.article_id = m.id
            WHERE NOT EXISTS (
                SELECT 1 FROM article_embedding e
                WHERE e.article_id = m.id
                  AND e.model_name = %(model_name)s
                  AND e.model_version = %(model_version)s
            )
            ORDER BY m.published_at NULLS LAST, m.id
            LIMIT %(limit)s;
            """,
            {"model_name": MODEL_NAME, "model_version": MODEL_VERSION, "limit": limit},
        )
        return list(cur.fetchall())


# _ensure_document is now the shared `ensure_document` from news_ingest.ml.document.


def _find_recent_embedding(conn, *, content_hash: str) -> int | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id
            FROM article_embedding_dedup
            WHERE content_hash = %(content_hash)s
              AND model_name = %(model_name)s
              AND model_version = %(model_version)s
              AND last_seen_at >= now() - interval '30 days'
            ORDER BY last_seen_at DESC
            LIMIT 1;
            """,
            {"content_hash": content_hash, "model_name": MODEL_NAME, "model_version": MODEL_VERSION},
        )
        row = cur.fetchone()
    return int(row["id"]) if row else None


def _touch_embedding(conn, embedding_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE article_embedding_dedup SET hit_count = hit_count + 1, last_seen_at = now() WHERE id = %(id)s;", {"id": embedding_id})


def _insert_embedding_dedup(conn, *, content_hash: str, embedding: list[float], token_count: int) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO article_embedding_dedup(content_hash, embedding, model_name, model_version, token_count)
            VALUES (%(content_hash)s, %(embedding)s::vector, %(model_name)s, %(model_version)s, %(token_count)s)
            ON CONFLICT (content_hash, model_name, model_version) DO UPDATE SET
                hit_count = article_embedding_dedup.hit_count + 1,
                last_seen_at = now()
            RETURNING id;
            """,
            {
                "content_hash": content_hash,
                "embedding": vector_literal(embedding),
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "token_count": token_count,
            },
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("embedding dedup insert did not return a row")
    return int(row["id"])


def _insert_mapping(conn, *, article_id: int, document_id: int, embedding_dedup_id: int, chunk) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO article_embedding(
                article_id, document_id, embedding_dedup_id, chunk_index, content_hash,
                char_start, char_end, token_count, model_name, model_version
            ) VALUES (
                %(article_id)s, %(document_id)s, %(embedding_dedup_id)s, %(chunk_index)s, %(content_hash)s,
                %(char_start)s, %(char_end)s, %(token_count)s, %(model_name)s, %(model_version)s
            )
            ON CONFLICT (article_id, content_hash, model_name, model_version) DO NOTHING
            RETURNING id;
            """,
            {
                "article_id": article_id,
                "document_id": document_id,
                "embedding_dedup_id": embedding_dedup_id,
                "chunk_index": chunk.index,
                "content_hash": chunk.content_hash,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "token_count": chunk.token_count,
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
            },
        )
        return cur.fetchone() is not None


def _process_article(conn, *, article: dict[str, Any], endpoint_url: str, batch_size: int, max_tokens: int, overlap_tokens: int, summary: EmbeddingSummary) -> None:
    document_text = get_article_document_text(article)
    if not document_text:
        summary.skipped_empty += 1
        return
    document_id = ensure_document(conn, article=article, document_text=document_text)
    chunks = fetch_token_chunks(endpoint_url, document_text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    summary.chunks_seen += len(chunks)

    pending = []
    for chunk in chunks:
        embedding_id = _find_recent_embedding(conn, content_hash=chunk.content_hash)
        if embedding_id is not None:
            _touch_embedding(conn, embedding_id)
            if _insert_mapping(conn, article_id=article["id"], document_id=document_id, embedding_dedup_id=embedding_id, chunk=chunk):
                summary.inserted_mappings += 1
            summary.reused_chunks += 1
        else:
            pending.append(chunk)

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        embeddings = fetch_embeddings(endpoint_url, [chunk.text for chunk in batch])
        if len(embeddings) != len(batch):
            raise RuntimeError(f"embedding endpoint returned {len(embeddings)} embeddings for {len(batch)} chunks")
        summary.endpoint_chunks += len(batch)
        for chunk, embedding in zip(batch, embeddings, strict=True):
            embedding_id = _insert_embedding_dedup(conn, content_hash=chunk.content_hash, embedding=embedding, token_count=chunk.token_count)
            if _insert_mapping(conn, article_id=article["id"], document_id=document_id, embedding_dedup_id=embedding_id, chunk=chunk):
                summary.inserted_mappings += 1
    conn.commit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate article chunk embeddings via the embedding service.")
    parser.add_argument("--limit", type=int, default=500, help="Maximum unprocessed articles to scan.")
    parser.add_argument("--batch-size", type=int, default=32, help="Chunks per embedding endpoint request.")
    parser.add_argument("--max-tokens", type=int, default=500, help="EmbeddingGemma tokenizer tokens per chunk.")
    parser.add_argument("--overlap-tokens", type=int, default=50, help="EmbeddingGemma tokenizer token overlap between chunks.")
    parser.add_argument("--endpoint-url", default=os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8001"), help="Embedding service base URL.")
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    settings = load_settings(require_newsapi=False, require_query=False)
    summary = process_embeddings(
        database_url=settings.database_url,
        endpoint_url=args.endpoint_url,
        limit=args.limit,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    logger.info("Finished embedding worker summary=%s", asdict(summary))
    print(asdict(summary))


if __name__ == "__main__":
    main()
