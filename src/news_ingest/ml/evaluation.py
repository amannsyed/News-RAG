from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from statistics import mean
from typing import Any

import psycopg
from psycopg.rows import dict_row

from news_ingest.config import load_settings
from news_ingest.db import ensure_schema
from news_ingest.ml.schema import ensure_ml_schema


CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class EvaluationResult:
    conversation_id: str
    created_at: str
    question: str | None
    answer_chars: int
    citation_count: int
    cited_marker_count: int
    referenced_citation_count: int
    citation_coverage: float
    web_citation_count: int
    retrieval_count: int
    cache_type: str | None
    score: float
    warnings: list[str]


def evaluate_recent_chats(*, database_url: str, limit: int = 25, user_id: str | None = None) -> dict[str, Any]:
    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        ensure_schema(conn)
        ensure_ml_schema(conn)
        rows = _load_recent_assistant_turns(conn, limit=limit, user_id=user_id)

    results = [_evaluate_row(row) for row in rows]
    return _summary(results)


def _load_recent_assistant_turns(conn, *, limit: int, user_id: str | None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"limit": limit, "user_id": user_id}
    user_filter = "AND COALESCE(a.payload->>'user_id', 'user_id') = %(user_id)s" if user_id else ""
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
                a.conversation_id,
                a.content AS answer,
                a.payload,
                a.created_at,
                (
                    SELECT u.content
                    FROM rag_conversations u
                    WHERE u.conversation_id = a.conversation_id
                      AND u.role = 'user'
                      AND u.created_at <= a.created_at
                    ORDER BY u.created_at DESC
                    LIMIT 1
                ) AS question
            FROM rag_conversations a
            WHERE a.role = 'assistant'
              {user_filter}
            ORDER BY a.created_at DESC
            LIMIT %(limit)s;
            """,
            params,
        )
        return [dict(row) for row in cur.fetchall()]


def _evaluate_row(row: dict[str, Any]) -> EvaluationResult:
    payload = row.get("payload") or {}
    answer = row.get("answer") or ""
    citations = payload.get("citations") or []
    retrieval = payload.get("retrieval") or {}
    retrieval_results = retrieval.get("results") or []
    citation_numbers = {int(match) for match in CITATION_RE.findall(answer)}
    valid_citation_indexes = {int(c.get("citation_index")) for c in citations if c.get("citation_index") is not None}
    referenced = citation_numbers & valid_citation_indexes
    citation_count = len(citations)
    citation_coverage = len(referenced) / citation_count if citation_count else 0.0
    web_citation_count = sum(1 for citation in citations if citation.get("source_type") == "web_search")
    warnings: list[str] = []

    if citation_count and not referenced:
        warnings.append("answer_has_sources_but_no_inline_citation_markers")
    if citation_numbers - valid_citation_indexes:
        warnings.append("answer_references_missing_citation_indexes")
    if len(answer) < 120:
        warnings.append("answer_is_short")
    if not retrieval_results:
        warnings.append("no_retrieval_results_in_payload")

    score_parts = [
        min(len(answer) / 600, 1.0),
        min(citation_count / 5, 1.0),
        citation_coverage,
        1.0 if retrieval_results else 0.0,
    ]
    score = round(mean(score_parts), 4)

    return EvaluationResult(
        conversation_id=row["conversation_id"],
        created_at=row["created_at"].isoformat() if hasattr(row.get("created_at"), "isoformat") else str(row.get("created_at")),
        question=row.get("question"),
        answer_chars=len(answer),
        citation_count=citation_count,
        cited_marker_count=len(citation_numbers),
        referenced_citation_count=len(referenced),
        citation_coverage=round(citation_coverage, 4),
        web_citation_count=web_citation_count,
        retrieval_count=len(retrieval_results),
        cache_type=(retrieval.get("cache") or {}).get("type") if isinstance(retrieval.get("cache"), dict) else None,
        score=score,
        warnings=warnings,
    )


def _summary(results: list[EvaluationResult]) -> dict[str, Any]:
    if not results:
        return {"count": 0, "avg_score": 0.0, "avg_citation_coverage": 0.0, "warning_counts": {}, "results": []}
    warning_counts: dict[str, int] = {}
    for result in results:
        for warning in result.warnings:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
    return {
        "count": len(results),
        "avg_score": round(mean(result.score for result in results), 4),
        "avg_citation_coverage": round(mean(result.citation_coverage for result in results), 4),
        "avg_citation_count": round(mean(result.citation_count for result in results), 2),
        "avg_retrieval_count": round(mean(result.retrieval_count for result in results), 2),
        "warning_counts": dict(sorted(warning_counts.items())),
        "results": [asdict(result) for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recent News RAG chat turns with lightweight deterministic checks.")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    database_url = args.database_url or load_settings(require_newsapi=False, require_query=False).database_url
    report = evaluate_recent_chats(database_url=database_url, limit=args.limit, user_id=args.user_id)
    print(json.dumps(report, indent=2 if args.pretty else None, default=str))


if __name__ == "__main__":
    main()
