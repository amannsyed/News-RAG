from __future__ import annotations

import argparse
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any

from psycopg.types.json import Jsonb

from news_ingest.config import load_settings
from news_ingest.db import connect, ensure_schema
from news_ingest.logging_config import configure_logging
from news_ingest.ml.chunk_client import fetch_token_chunks
from news_ingest.ml.http_client import post_json
from news_ingest.ml.ner import NER_LABELS, EntityMention, deduplicate_overlaps, entity_from_payload, normalize_entity_text, salience_score
from news_ingest.ml.schema import ensure_ml_schema
from news_ingest.ml.document import ensure_document, get_article_document_text


logger = logging.getLogger(__name__)
MODEL_NAME = "urchade/gliner_multi-v2.1"
MODEL_VERSION = "v1"
DOCUMENT_SCOPE = "ml-v1"
LABELS = NER_LABELS


@dataclass
class NerSummary:
    articles_seen: int = 0
    chunks_seen: int = 0
    endpoint_chunks: int = 0
    reused_chunks: int = 0
    entities_seen: int = 0
    entities_inserted: int = 0
    skipped_empty: int = 0
    errors: int = 0


def fetch_entities(endpoint_url: str, texts: list[str], *, threshold: float) -> list[list[dict[str, Any]]]:
    if not texts:
        return []
    payload = post_json(
        endpoint_url.rstrip("/") + "/ner",
        {"texts": texts, "labels": [label.lower() for label in LABELS], "threshold": threshold},
        timeout=180,
    )
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("NER endpoint response missing results list")
    return results


def process_ner(*, database_url: str, endpoint_url: str, chunk_endpoint_url: str, limit: int = 500, batch_size: int = 16, max_tokens: int = 500, overlap_tokens: int = 50, threshold: float = 0.5) -> NerSummary:
    summary = NerSummary()
    with connect(database_url) as conn:
        ensure_schema(conn)
        ensure_ml_schema(conn)
        articles = _load_unprocessed_articles(conn, limit=limit)
        summary.articles_seen = len(articles)
        for article in articles:
            try:
                _process_article(conn, article=article, endpoint_url=endpoint_url, chunk_endpoint_url=chunk_endpoint_url, batch_size=batch_size, max_tokens=max_tokens, overlap_tokens=overlap_tokens, threshold=threshold, summary=summary)
            except Exception:
                conn.rollback()
                summary.errors += 1
                logger.exception("Failed NER processing article_id=%s", article.get("id"))
    return summary


def _load_unprocessed_articles(conn, *, limit: int) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT m.id, m.title, m.description, c.content
            FROM article_metadata m
            JOIN article_contents c ON c.article_id = m.id
            WHERE NOT EXISTS (
                SELECT 1 FROM article_entities e
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


def _find_ner_cache(conn, *, content_hash: str) -> tuple[int, list[dict[str, Any]]] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, payload
            FROM ner_content_dedup
            WHERE content_hash = %(content_hash)s
              AND model_name = %(model_name)s
              AND model_version = %(model_version)s
              AND labels = %(labels)s
            LIMIT 1;
            """,
            {"content_hash": content_hash, "model_name": MODEL_NAME, "model_version": MODEL_VERSION, "labels": list(LABELS)},
        )
        row = cur.fetchone()
    if not row:
        return None
    payload = row["payload"] or []
    return int(row["id"]), list(payload)


def _touch_ner_cache(conn, cache_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE ner_content_dedup SET hit_count = hit_count + 1, last_seen_at = now() WHERE id = %(id)s;", {"id": cache_id})


def _insert_ner_cache(conn, *, content_hash: str, payload: list[dict[str, Any]]) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ner_content_dedup(content_hash, model_name, model_version, labels, payload)
            VALUES (%(content_hash)s, %(model_name)s, %(model_version)s, %(labels)s, %(payload)s)
            ON CONFLICT (content_hash, model_name, model_version, labels) DO UPDATE SET
                hit_count = ner_content_dedup.hit_count + 1,
                last_seen_at = now()
            RETURNING id;
            """,
            {"content_hash": content_hash, "model_name": MODEL_NAME, "model_version": MODEL_VERSION, "labels": list(LABELS), "payload": Jsonb(payload)},
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("NER cache insert did not return a row")
    return int(row["id"])


def _entity_type_id(conn, label: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM entity_types WHERE label = %(label)s;", {"label": label})
        row = cur.fetchone()
    if row is None:
        raise RuntimeError(f"missing entity type {label}")
    return int(row["id"])


def _alias_id(conn, *, entity_type_id: int, text: str, normalized_text: str) -> int:
    with conn.cursor() as cur:
        # Use the %% operator so Postgres can use the GIN trigram index
        # (idx_entity_aliases_trgm).  A bare similarity() > threshold in the
        # WHERE clause bypasses the index and causes a sequential scan.
        cur.execute(
            """
            SELECT id, canonical_name
            FROM entity_aliases
            WHERE entity_type_id = %(entity_type_id)s
              AND normalized_alias %% %(normalized_text)s
            ORDER BY similarity(normalized_alias, %(normalized_text)s) DESC
            LIMIT 1;
            """,
            {"entity_type_id": entity_type_id, "normalized_text": normalized_text},
        )
        row = cur.fetchone()
        if row:
            return int(row["id"])
        cur.execute(
            """
            INSERT INTO entity_aliases(entity_type_id, alias, normalized_alias, canonical_name)
            VALUES (%(entity_type_id)s, %(alias)s, %(normalized_alias)s, %(canonical_name)s)
            ON CONFLICT (entity_type_id, normalized_alias) DO UPDATE SET updated_at = now()
            RETURNING id;
            """,
            {"entity_type_id": entity_type_id, "alias": text, "normalized_alias": normalized_text, "canonical_name": text},
        )
        inserted = cur.fetchone()
    if inserted is None:
        raise RuntimeError("alias insert did not return a row")
    return int(inserted["id"])


def _insert_entity(conn, *, article_id: int, document_id: int, ner_dedup_id: int, entity: EntityMention, article_length: int, entity_salience: float) -> bool:
    normalized = normalize_entity_text(entity.text)
    if not normalized:
        return False
    entity_type_id = _entity_type_id(conn, entity.label)
    alias_id = _alias_id(conn, entity_type_id=entity_type_id, text=entity.text, normalized_text=normalized)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO article_entities(
                article_id, document_id, entity_type_id, alias_id, ner_dedup_id,
                entity_text, normalized_text, char_start, char_end, confidence,
                salience, model_name, model_version
            ) VALUES (
                %(article_id)s, %(document_id)s, %(entity_type_id)s, %(alias_id)s, %(ner_dedup_id)s,
                %(entity_text)s, %(normalized_text)s, %(char_start)s, %(char_end)s, %(confidence)s,
                %(salience)s, %(model_name)s, %(model_version)s
            )
            ON CONFLICT (article_id, entity_type_id, normalized_text, char_start, char_end, model_name, model_version) DO NOTHING
            RETURNING id;
            """,
            {
                "article_id": article_id,
                "document_id": document_id,
                "entity_type_id": entity_type_id,
                "alias_id": alias_id,
                "ner_dedup_id": ner_dedup_id,
                "entity_text": entity.text,
                "normalized_text": normalized,
                "char_start": entity.start,
                "char_end": entity.end,
                "confidence": entity.confidence,
                "salience": entity_salience,
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
            },
        )
        return cur.fetchone() is not None


def _process_article(conn, *, article: dict[str, Any], endpoint_url: str, chunk_endpoint_url: str, batch_size: int, max_tokens: int, overlap_tokens: int, threshold: float, summary: NerSummary) -> None:
    document_text = get_article_document_text(article)
    if not document_text:
        summary.skipped_empty += 1
        return
    document_id = ensure_document(conn, article=article, document_text=document_text)
    chunks = fetch_token_chunks(chunk_endpoint_url, document_text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    summary.chunks_seen += len(chunks)

    chunk_payloads: list[tuple[int, Any, list[dict[str, Any]]]] = []
    pending = []
    for chunk in chunks:
        cached = _find_ner_cache(conn, content_hash=chunk.content_hash)
        if cached is not None:
            cache_id, payload = cached
            _touch_ner_cache(conn, cache_id)
            chunk_payloads.append((cache_id, chunk, payload))
            summary.reused_chunks += 1
        else:
            pending.append(chunk)

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        results = fetch_entities(endpoint_url, [chunk.text for chunk in batch], threshold=threshold)
        if len(results) != len(batch):
            raise RuntimeError(f"NER endpoint returned {len(results)} result sets for {len(batch)} chunks")
        summary.endpoint_chunks += len(batch)
        for chunk, payload in zip(batch, results, strict=True):
            cache_id = _insert_ner_cache(conn, content_hash=chunk.content_hash, payload=list(payload))
            chunk_payloads.append((cache_id, chunk, list(payload)))

    mentions = []
    mention_cache_ids = []
    for cache_id, chunk, payload in chunk_payloads:
        for raw_entity in payload:
            entity = entity_from_payload(raw_entity, chunk_offset=chunk.char_start)
            if entity is not None:
                mentions.append(entity)
                mention_cache_ids.append(cache_id)

    # Keep cache id aligned after overlap removal by choosing the first matching mention/cache pair.
    deduped = deduplicate_overlaps(mentions)
    salience_by_key: dict[tuple[str, str], float] = {}
    for entity in deduped:
        key = (entity.label, normalize_entity_text(entity.text))
        salience_by_key[key] = salience_by_key.get(key, 0.0) + salience_score(confidence=entity.confidence, position=entity.start, article_length=len(document_text))

    summary.entities_seen += len(deduped)
    for entity in deduped:
        try:
            cache_id = next((cid for cid, original in zip(mention_cache_ids, mentions, strict=True) if original == entity), None)
        except ValueError:
            cache_id = None
        if cache_id is None:
            cache_id = chunk_payloads[0][0] if chunk_payloads else None
        key = (entity.label, normalize_entity_text(entity.text))
        if _insert_entity(
            conn,
            article_id=article["id"],
            document_id=document_id,
            ner_dedup_id=int(cache_id),
            entity=entity,
            article_length=len(document_text),
            entity_salience=salience_by_key[key],
        ):
            summary.entities_inserted += 1
    conn.commit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract article entities via the NER service.")
    parser.add_argument("--limit", type=int, default=500, help="Maximum unprocessed articles to scan.")
    parser.add_argument("--batch-size", type=int, default=16, help="Chunks per NER endpoint request.")
    parser.add_argument("--max-tokens", type=int, default=500, help="EmbeddingGemma tokenizer tokens per chunk.")
    parser.add_argument("--overlap-tokens", type=int, default=50, help="EmbeddingGemma tokenizer token overlap between chunks.")
    parser.add_argument("--threshold", type=float, default=0.5, help="GLiNER confidence threshold.")
    parser.add_argument("--endpoint-url", default=os.getenv("NER_SERVICE_URL", "http://localhost:8002"), help="NER service base URL.")
    parser.add_argument("--chunk-endpoint-url", default=os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8001"), help="Embedding service base URL used for EmbeddingGemma tokenizer chunking.")
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    settings = load_settings(require_newsapi=False, require_query=False)
    summary = process_ner(
        database_url=settings.database_url,
        endpoint_url=args.endpoint_url,
        chunk_endpoint_url=args.chunk_endpoint_url,
        limit=args.limit,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        threshold=args.threshold,
    )
    logger.info("Finished NER worker summary=%s", asdict(summary))
    print(asdict(summary))


if __name__ == "__main__":
    main()
