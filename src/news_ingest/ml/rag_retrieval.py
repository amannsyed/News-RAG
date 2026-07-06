from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from news_ingest.db import connect, ensure_schema
from news_ingest.ml.embedding_worker import vector_literal
from news_ingest.ml.http_client import post_json
from news_ingest.ml.rag_cache import cache_fingerprint, get_exact_cache, get_semantic_cache, put_cache, to_jsonable
from news_ingest.ml.ner import normalize_entity_text
from news_ingest.ml.schema import ensure_ml_schema


DEFAULT_RRF_K = 60
DEFAULT_LIMIT = 10
MAX_OVERFETCH = 100


@dataclass(frozen=True)
class RetrievalWeights:
    vector: float = 0.6
    full_text: float = 0.2
    entity: float = 0.2


@dataclass(frozen=True)
class StreamHit:
    article_id: int
    rank: int
    score: float
    stream: str
    title: str | None = None
    url: str | None = None
    provider: str | None = None
    published_at: str | None = None
    snippet: str | None = None
    chunk_id: int | None = None
    document_id: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    entity_text: str | None = None
    entity_type: str | None = None


@dataclass
class FusedResult:
    article_id: int
    rrf_score: float
    title: str | None
    url: str | None
    provider: str | None
    published_at: str | None
    stream_scores: dict[str, float] = field(default_factory=dict)
    stream_ranks: dict[str, int] = field(default_factory=dict)
    matches: list[StreamHit] = field(default_factory=list)


def fetch_query_embedding(endpoint_url: str, query: str) -> list[float]:
    payload = post_json(endpoint_url.rstrip("/") + "/embed", {"texts": [query], "input_type": "query"}, timeout=120)
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings or not isinstance(embeddings[0], list):
        raise RuntimeError("embedding endpoint response missing query embedding")
    return embeddings[0]


def normalize_weights(weights: RetrievalWeights) -> RetrievalWeights:
    total = weights.vector + weights.full_text + weights.entity
    if total <= 0:
        return RetrievalWeights()
    return RetrievalWeights(vector=weights.vector / total, full_text=weights.full_text / total, entity=weights.entity / total)


def dynamic_weights(*, base: RetrievalWeights, has_entities: bool, request_override: RetrievalWeights | None = None) -> RetrievalWeights:
    if request_override is not None:
        return normalize_weights(request_override)
    if has_entities:
        return normalize_weights(RetrievalWeights(vector=0.45, full_text=0.20, entity=0.35))
    return normalize_weights(RetrievalWeights(vector=0.70, full_text=0.25, entity=0.05))


def rrf_fuse(streams: dict[str, list[StreamHit]], *, weights: RetrievalWeights, k: int = DEFAULT_RRF_K, limit: int = DEFAULT_LIMIT) -> list[FusedResult]:
    by_article: dict[int, FusedResult] = {}
    stream_weights = {"vector": weights.vector, "full_text": weights.full_text, "entity": weights.entity}
    for stream_name, hits in streams.items():
        weight = stream_weights.get(stream_name, 0.0)
        if weight <= 0:
            continue
        seen_articles: set[int] = set()
        for hit in hits:
            if hit.article_id in seen_articles:
                continue
            seen_articles.add(hit.article_id)
            contribution = weight * (1.0 / (k + hit.rank))
            result = by_article.get(hit.article_id)
            if result is None:
                result = FusedResult(
                    article_id=hit.article_id,
                    rrf_score=0.0,
                    title=hit.title,
                    url=hit.url,
                    provider=hit.provider,
                    published_at=hit.published_at,
                )
                by_article[hit.article_id] = result
            result.rrf_score += contribution
            result.stream_scores[stream_name] = hit.score
            result.stream_ranks[stream_name] = hit.rank
            result.matches.append(hit)
    return sorted(by_article.values(), key=lambda item: item.rrf_score, reverse=True)[:limit]


def overfetch_limit(limit: int) -> int:
    return max(1, min(MAX_OVERFETCH, limit * 2))


def search_articles(
    *,
    database_url: str,
    embedding_endpoint_url: str,
    query: str,
    limit: int = DEFAULT_LIMIT,
    weights: RetrievalWeights | None = None,
    rrf_k: int = DEFAULT_RRF_K,
) -> dict[str, Any]:
    base_weights = RetrievalWeights(
        vector=float(os.getenv("RAG_VECTOR_WEIGHT", "0.6")),
        full_text=float(os.getenv("RAG_FULL_TEXT_WEIGHT", "0.2")),
        entity=float(os.getenv("RAG_ENTITY_WEIGHT", "0.2")),
    )
    with connect(database_url) as conn:
        ensure_schema(conn)
        ensure_ml_schema(conn)
        entity_candidates = candidate_entities(conn, query=query, limit=20)
        selected_weights = dynamic_weights(base=base_weights, has_entities=bool(entity_candidates), request_override=weights)
        weights_dict = {"vector": selected_weights.vector, "full_text": selected_weights.full_text, "entity": selected_weights.entity}
        fingerprint = cache_fingerprint(limit=limit, rrf_k=rrf_k, weights=weights_dict)

        cached = get_exact_cache(conn, query=query, fingerprint=fingerprint)
        if cached is not None:
            cached["cache"] = {"hit": True, "type": "exact"}
            return cached

        query_embedding = fetch_query_embedding(embedding_endpoint_url, query)
        cached = get_semantic_cache(conn, embedding=query_embedding, fingerprint=fingerprint, threshold=float(os.getenv("RAG_SEMANTIC_CACHE_THRESHOLD", "0.95")))
        if cached is not None:
            return cached

        fetch_limit = overfetch_limit(limit)
        streams = {
            "vector": vector_search(conn, query_embedding=query_embedding, limit=fetch_limit),
            "full_text": full_text_search(conn, query=query, limit=fetch_limit),
            "entity": entity_search(conn, query=query, candidate_alias_ids=[item["alias_id"] for item in entity_candidates], limit=fetch_limit),
        }
        fused = rrf_fuse(streams, weights=selected_weights, k=rrf_k, limit=limit)
        response = {
            "query": query,
            "limit": limit,
            "rrf_k": rrf_k,
            "weights": weights_dict,
            "entity_candidates": entity_candidates,
            "results": fused,
            "cache": {"hit": False, "type": None},
        }
        put_cache(conn, query=query, embedding=query_embedding, fingerprint=fingerprint, response=to_jsonable(response), ttl_seconds=int(os.getenv("RAG_CACHE_TTL_SECONDS", "3600")))
        return response


def candidate_entities(conn, *, query: str, limit: int) -> list[dict[str, Any]]:
    normalized = normalize_entity_text(query)
    if not normalized:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ea.id AS alias_id, et.label, ea.alias, ea.normalized_alias,
                   GREATEST(similarity(ea.normalized_alias, %(query)s), similarity(ea.alias, %(raw_query)s)) AS score
            FROM entity_aliases ea
            JOIN entity_types et ON et.id = ea.entity_type_id
            WHERE ea.normalized_alias %% %(query)s
               OR %(query)s ILIKE '%%' || ea.normalized_alias || '%%'
               OR ea.normalized_alias ILIKE '%%' || %(query)s || '%%'
            ORDER BY score DESC, length(ea.normalized_alias) DESC
            LIMIT %(limit)s;
            """,
            {"query": normalized, "raw_query": query, "limit": limit},
        )
        return [dict(row) for row in cur.fetchall()]


def vector_search(conn, *, query_embedding: list[float], limit: int) -> list[StreamHit]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.article_id, e.id AS chunk_id, e.document_id, e.char_start, e.char_end,
                   m.title, m.url, m.provider, m.published_at::text AS published_at,
                   substring(d.document_text FROM e.char_start + 1 FOR e.char_end - e.char_start) AS snippet,
                   1 - (dedup.embedding <=> %(embedding)s::vector) AS score
            FROM article_embedding e
            JOIN article_embedding_dedup dedup ON dedup.id = e.embedding_dedup_id
            JOIN article_ml_documents d ON d.id = e.document_id
            JOIN article_metadata m ON m.id = e.article_id
            ORDER BY dedup.embedding <=> %(embedding)s::vector
            LIMIT %(limit)s;
            """,
            {"embedding": vector_literal(query_embedding), "limit": limit},
        )
        rows = cur.fetchall()
    return [_hit_from_row(row, rank=index + 1, stream="vector") for index, row in enumerate(rows)]


def full_text_search(conn, *, query: str, limit: int) -> list[StreamHit]:
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH q AS (SELECT websearch_to_tsquery('english', %(query)s) AS tsq)
            SELECT d.article_id, d.id AS document_id,
                   m.title, m.url, m.provider, m.published_at::text AS published_at,
                   ts_headline('english', d.document_text, q.tsq, 'MaxWords=45, MinWords=12, ShortWord=3') AS snippet,
                   ts_rank_cd(to_tsvector('english', d.document_text), q.tsq) AS score
            FROM article_ml_documents d
            JOIN article_metadata m ON m.id = d.article_id
            CROSS JOIN q
            WHERE q.tsq @@ to_tsvector('english', d.document_text)
            ORDER BY score DESC, m.published_at DESC NULLS LAST
            LIMIT %(limit)s;
            """,
            {"query": query, "limit": limit},
        )
        rows = cur.fetchall()
    return [_hit_from_row(row, rank=index + 1, stream="full_text") for index, row in enumerate(rows)]


def entity_search(conn, *, query: str, candidate_alias_ids: list[int], limit: int) -> list[StreamHit]:
    """Return one result per article, using the highest-salience matching entity.

    Two deliberate changes from the original:

    1. **Per-article deduplication at DB level** — DISTINCT ON (ae.article_id) ensures
       that if "Micron" appears 10 times in one article we still return that article once,
       ranked by its highest-salience entity mention.  Previously this deduplication only
       happened inside rrf_fuse, which meant entity_search returned O(mentions) rows and
       passed many redundant hits into the fusion step.

    2. **Full document text as snippet** — instead of extracting only the entity span
       (e.g. just the string "Micron"), we return the entire article ML document so the
       LLM receives real context when this stream is the sole or primary match for an article.
    """
    if not candidate_alias_ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT article_id, document_id, title, url, provider, published_at,
                   snippet, entity_text, entity_type, score
            FROM (
                SELECT DISTINCT ON (ae.article_id)
                    ae.article_id,
                    ae.document_id,
                    m.title,
                    m.url,
                    m.provider,
                    m.published_at::text AS published_at,
                    d.document_text      AS snippet,
                    ae.entity_text,
                    et.label             AS entity_type,
                    ae.salience          AS score
                FROM article_entities ae
                JOIN entity_types et        ON et.id  = ae.entity_type_id
                JOIN article_metadata m     ON m.id   = ae.article_id
                JOIN article_ml_documents d ON d.id   = ae.document_id
                WHERE ae.alias_id = ANY(%(alias_ids)s)
                ORDER BY ae.article_id, ae.salience DESC
            ) top_per_article
            ORDER BY score DESC, published_at DESC NULLS LAST
            LIMIT %(limit)s;
            """,
            {"alias_ids": candidate_alias_ids, "limit": limit},
        )
        rows = cur.fetchall()
    return [_hit_from_row(row, rank=index + 1, stream="entity") for index, row in enumerate(rows)]


def _hit_from_row(row: dict[str, Any], *, rank: int, stream: str) -> StreamHit:
    return StreamHit(
        article_id=int(row["article_id"]),
        rank=rank,
        score=float(row.get("score") or 0.0),
        stream=stream,
        title=row.get("title"),
        url=row.get("url"),
        provider=row.get("provider"),
        published_at=row.get("published_at"),
        snippet=_clean_snippet(row.get("snippet")),
        chunk_id=int(row["chunk_id"]) if row.get("chunk_id") is not None else None,
        document_id=int(row["document_id"]) if row.get("document_id") is not None else None,
        char_start=int(row["char_start"]) if row.get("char_start") is not None else None,
        char_end=int(row["char_end"]) if row.get("char_end") is not None else None,
        entity_text=row.get("entity_text"),
        entity_type=row.get("entity_type"),
    )


def _clean_snippet(value: str | None) -> str | None:
    if value is None:
        return None
    return re.sub(r"\s+", " ", value).strip()
