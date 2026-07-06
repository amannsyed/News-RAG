from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from psycopg.types.json import Jsonb

from news_ingest.ml.embedding_worker import vector_literal


def stable_json_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def query_hash(query: str) -> str:
    return hashlib.sha256(" ".join(query.lower().split()).encode("utf-8")).hexdigest()


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    return value


def cache_fingerprint(*, limit: int, rrf_k: int, weights: dict[str, float]) -> str:
    return stable_json_hash({"limit": limit, "rrf_k": rrf_k, "weights": weights})


def get_exact_cache(conn, *, query: str, fingerprint: str) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, response
            FROM rag_search_cache
            WHERE query_hash = %(query_hash)s
              AND request_fingerprint = %(fingerprint)s
              AND expires_at > now()
            ORDER BY created_at DESC
            LIMIT 1;
            """,
            {"query_hash": query_hash(query), "fingerprint": fingerprint},
        )
        row = cur.fetchone()
        if row:
            cur.execute("UPDATE rag_search_cache SET hit_count = hit_count + 1, last_hit_at = now() WHERE id = %(id)s;", {"id": row["id"]})
            conn.commit()
            return dict(row["response"])
    return None


def get_semantic_cache(conn, *, embedding: list[float], fingerprint: str, threshold: float = 0.95) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, response, 1 - (query_embedding <=> %(embedding)s::vector) AS similarity
            FROM rag_search_cache
            WHERE request_fingerprint = %(fingerprint)s
              AND expires_at > now()
            ORDER BY query_embedding <=> %(embedding)s::vector
            LIMIT 1;
            """,
            {"embedding": vector_literal(embedding), "fingerprint": fingerprint},
        )
        row = cur.fetchone()
        if row and float(row["similarity"] or 0.0) >= threshold:
            response = dict(row["response"])
            response["cache"] = {"hit": True, "type": "semantic", "similarity": float(row["similarity"])}
            cur.execute("UPDATE rag_search_cache SET hit_count = hit_count + 1, last_hit_at = now() WHERE id = %(id)s;", {"id": row["id"]})
            conn.commit()
            return response
    return None


def put_cache(conn, *, query: str, embedding: list[float], fingerprint: str, response: dict[str, Any], ttl_seconds: int = 3600) -> None:
    expires_at = datetime.now(UTC) + timedelta(seconds=ttl_seconds)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO rag_search_cache(query_hash, query, query_embedding, request_fingerprint, response, expires_at)
            VALUES (%(query_hash)s, %(query)s, %(embedding)s::vector, %(fingerprint)s, %(response)s, %(expires_at)s)
            ON CONFLICT (query_hash, request_fingerprint) DO UPDATE SET
                query_embedding = EXCLUDED.query_embedding,
                response = EXCLUDED.response,
                expires_at = EXCLUDED.expires_at;
            """,
            {
                "query_hash": query_hash(query),
                "query": query,
                "embedding": vector_literal(embedding),
                "fingerprint": fingerprint,
                "response": Jsonb(to_jsonable(response)),
                "expires_at": expires_at,
            },
        )
    conn.commit()
