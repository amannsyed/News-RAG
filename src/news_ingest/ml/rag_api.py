from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field, model_validator

from news_ingest.config import load_settings
from news_ingest.db import connect, ensure_schema
from news_ingest.ml.evaluation import evaluate_recent_chats
from news_ingest.ml.claude_vertex import generate_chat_answer, generate_chat_summary, generate_follow_up_questions, load_chat_settings, with_model_override
from news_ingest.ml.rag_cache import to_jsonable
from news_ingest.ml.rag_retrieval import RetrievalWeights, search_articles
from news_ingest.ml.rag_security import check_rate_limit, require_auth
from news_ingest.ml.schema import ensure_ml_schema
from news_ingest.ml.text import extract_keyword_context, extract_query_keywords
from news_ingest.ml.web_search import fetch_and_index_web_search


logger = logging.getLogger(__name__)
DEFAULT_USER_ID = "user_id"
ThinkingLevel = Literal["low", "medium", "max"]

# ---------------------------------------------------------------------------
# Connection pool — initialised once in the lifespan hook.
# search_articles() opens its own short-lived connection (it may run on a
# separate thread/worker in future); all other DB work inside this API uses
# the pool so we avoid creating a new OS-level connection per request.
# ---------------------------------------------------------------------------
_pool: ConnectionPool | None = None


def _database_url() -> str:
    return os.getenv("DATABASE_URL") or load_settings(require_newsapi=False, require_query=False).database_url


def _embedding_url() -> str:
    return os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8001")


def _get_pool() -> ConnectionPool:
    if _pool is None:
        raise RuntimeError("Connection pool is not initialised — did the lifespan hook run?")
    return _pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialise pool, run schema migrations, log warnings."""
    global _pool

    database_url = _database_url()

    # Warn loudly when no auth token is configured — the API is then open.
    if not os.getenv("RAG_API_TOKEN", "").strip():
        logger.warning(
            "RAG_API_TOKEN is not set — all endpoints are unauthenticated. "
            "Set RAG_API_TOKEN in your environment before exposing this service."
        )

    # Warn about multi-worker rate-limit limitation.
    if not os.getenv("RAG_RATE_LIMIT_REDIS_URL", "").strip():
        logger.warning(
            "Rate-limit state is stored in process memory. "
            "If you run multiple uvicorn workers the effective rate limit is "
            "multiplied by the worker count. Run with --workers 1 or set "
            "RAG_RATE_LIMIT_REDIS_URL to enable a shared store."
        )

    _pool = ConnectionPool(
        database_url,
        min_size=2,
        max_size=10,
        kwargs={"row_factory": dict_row},
        open=True,
    )

    # Run schema migrations once at startup rather than on every request.
    with _pool.connection() as conn:
        ensure_schema(conn)
        ensure_ml_schema(conn)

    logger.info("RAG API startup complete — pool ready, schema confirmed.")
    yield

    _pool.close()
    logger.info("RAG API shutdown — pool closed.")


app = FastAPI(title="News RAG Retrieval API", lifespan=lifespan)


class WeightOverride(BaseModel):
    vector: float | None = Field(default=None, ge=0)
    full_text: float | None = Field(default=None, ge=0)
    entity: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def has_any_weight(self):
        if self.vector is None and self.full_text is None and self.entity is None:
            raise ValueError("at least one weight must be set")
        return self

    def to_weights(self) -> RetrievalWeights:
        return RetrievalWeights(
            vector=self.vector if self.vector is not None else 0.0,
            full_text=self.full_text if self.full_text is not None else 0.0,
            entity=self.entity if self.entity is not None else 0.0,
        )


class UnifiedSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=50)
    rrf_k: int = Field(default=60, ge=1, le=500)
    weights: WeightOverride | None = None


class ConversationRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=120)
    user_id: str = Field(default=DEFAULT_USER_ID, min_length=1, max_length=120)
    limit: int = Field(default=20, ge=1, le=25)
    rrf_k: int = Field(default=60, ge=1, le=500)
    weights: WeightOverride | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=120)
    user_id: str = Field(default=DEFAULT_USER_ID, min_length=1, max_length=120)
    limit: int = Field(default=20, ge=1, le=25)
    rrf_k: int = Field(default=60, ge=1, le=500)
    weights: WeightOverride | None = None
    web_search: bool = False
    web_search_limit: int = Field(default=20, ge=1, le=20)
    model: str | None = Field(default=None, max_length=200)
    thinking_enabled: bool = False
    thinking_level: ThinkingLevel = "medium"


class MatchResponse(BaseModel):
    stream: str
    rank: int
    score: float
    snippet: str | None = None
    chunk_id: int | None = None
    document_id: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    entity_text: str | None = None
    entity_type: str | None = None


class SearchResultResponse(BaseModel):
    article_id: int
    rrf_score: float
    title: str | None = None
    url: str | None = None
    provider: str | None = None
    published_at: str | None = None
    stream_scores: dict[str, float]
    stream_ranks: dict[str, int]
    matches: list[MatchResponse]


class SearchResponse(BaseModel):
    query: str
    limit: int
    rrf_k: int
    weights: dict[str, float]
    entity_candidates: list[dict[str, Any]]
    results: list[SearchResultResponse]
    cache: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: list[dict[str, Any]]
    references: list[dict[str, Any]]
    retrieval: SearchResponse
    model: str
    follow_up_questions: list[str] = Field(default_factory=list)
    conversation_summary: str | None = None
    thinking_enabled: bool = False
    thinking_level: str | None = None


class ConversationSummary(BaseModel):
    id: str
    title: str
    date: str


class MonitoringSummaryResponse(BaseModel):
    window_hours: int
    total_requests: int
    errors: int
    error_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    avg_citations: float
    avg_retrieval_count: float
    by_endpoint: list[dict[str, Any]]
    recent_errors: list[dict[str, Any]]


class MessageResponse(BaseModel):
    role: str
    content: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)


class DeleteConversationResponse(BaseModel):
    conversation_id: str
    deleted_messages: int
    deleted_summary: bool


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "service": "rag-api", "embedding_service_url": _embedding_url()}




@app.get("/monitoring/summary", response_model=MonitoringSummaryResponse)
def monitoring_summary(http_request: Request, window_hours: int = Query(24, ge=1, le=24 * 30), identity: str = Depends(require_auth)) -> MonitoringSummaryResponse:
    check_rate_limit(http_request, identity)
    with _get_pool().connection() as conn:
        summary = _load_monitoring_summary(conn, window_hours=window_hours)
    return MonitoringSummaryResponse(**summary)




@app.get("/evaluation/recent")
def evaluation_recent(http_request: Request, limit: int = Query(25, ge=1, le=200), user_id: str | None = Query(default=None, min_length=1, max_length=120), identity: str = Depends(require_auth)) -> dict[str, Any]:
    check_rate_limit(http_request, identity)
    return evaluate_recent_chats(database_url=_database_url(), limit=limit, user_id=user_id)


@app.post("/search", response_model=SearchResponse)
def search(request: UnifiedSearchRequest, http_request: Request, identity: str = Depends(require_auth)) -> SearchResponse:
    check_rate_limit(http_request, identity)
    started_at = time.perf_counter()
    try:
        payload = search_articles(
            database_url=_database_url(),
            embedding_endpoint_url=_embedding_url(),
            query=request.query,
            limit=request.limit,
            weights=request.weights.to_weights() if request.weights else None,
            rrf_k=request.rrf_k,
        )
        response = _search_response_from_payload(payload)
        _record_rag_metric(
            endpoint="/search",
            started_at=started_at,
            query=request.query,
            status="ok",
            retrieval_count=len(response.results),
            citation_count=0,
            cache_type=(response.cache or {}).get("type") if response.cache else None,
            payload={"identity": identity, "limit": request.limit, "weights": response.weights},
        )
        return response
    except Exception as exc:
        _record_rag_metric(endpoint="/search", started_at=started_at, query=request.query, status="error", error=str(exc), payload={"identity": identity})
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/conversations", response_model=list[ConversationSummary])
def get_conversations(http_request: Request, user_id: str = Query(DEFAULT_USER_ID, min_length=1, max_length=120), identity: str = Depends(require_auth)) -> list[ConversationSummary]:
    check_rate_limit(http_request, identity)
    with _get_pool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    conversation_id,
                    MAX(created_at) as last_activity,
                    (
                        SELECT content
                        FROM rag_conversations rc2
                        WHERE rc2.conversation_id = rc.conversation_id
                          AND role = 'user'
                          AND COALESCE(rc2.payload->>'user_id', %(default_user_id)s) = %(user_id)s
                        ORDER BY created_at ASC
                        LIMIT 1
                    ) as title
                FROM rag_conversations rc
                WHERE COALESCE(payload->>'user_id', %(default_user_id)s) = %(user_id)s
                GROUP BY conversation_id
                ORDER BY last_activity DESC
            """, {"user_id": user_id, "default_user_id": DEFAULT_USER_ID})
            rows = cur.fetchall()
            
    summaries = []
    for row in rows:
        title = row["title"] or "New Chat"
        if len(title) > 30:
            title = title[:30] + "..."
        date_str = row["last_activity"].strftime("%b %d, %Y") if row["last_activity"] else "Recently"
        summaries.append(ConversationSummary(id=row["conversation_id"], title=title, date=date_str))
        
    return summaries


@app.get("/conversations/{conversation_id}", response_model=list[MessageResponse])
def get_conversation(conversation_id: str, http_request: Request, user_id: str = Query(DEFAULT_USER_ID, min_length=1, max_length=120), identity: str = Depends(require_auth)) -> list[MessageResponse]:
    check_rate_limit(http_request, identity)
    with _get_pool().connection() as conn:
        _ensure_conversation_access(conn, conversation_id=conversation_id, user_id=user_id)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT role, content, payload
                FROM rag_conversations
                WHERE conversation_id = %(conversation_id)s
                ORDER BY created_at ASC
            """, {"conversation_id": conversation_id})
            rows = cur.fetchall()
            
    messages = []
    for row in rows:
        payload = row["payload"] or {}
        citations = payload.get("citations", [])
        follow_ups = payload.get("follow_up_questions", [])
        messages.append(MessageResponse(role=row["role"], content=row["content"], citations=citations, follow_up_questions=follow_ups))
    return messages


@app.delete("/conversations/{conversation_id}", response_model=DeleteConversationResponse)
def delete_conversation(conversation_id: str, http_request: Request, user_id: str = Query(DEFAULT_USER_ID, min_length=1, max_length=120), identity: str = Depends(require_auth)) -> DeleteConversationResponse:
    check_rate_limit(http_request, identity)
    with _get_pool().connection() as conn:
        _ensure_conversation_access(conn, conversation_id=conversation_id, user_id=user_id)
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM rag_conversations
                WHERE conversation_id = %(conversation_id)s
                  AND COALESCE(payload->>'user_id', %(default_user_id)s) = %(user_id)s;
                """,
                {"conversation_id": conversation_id, "user_id": user_id, "default_user_id": DEFAULT_USER_ID},
            )
            deleted_messages = cur.rowcount or 0
            cur.execute(
                """
                DELETE FROM rag_conversation_summaries
                WHERE conversation_id = %(conversation_id)s
                  AND user_id = %(user_id)s;
                """,
                {"conversation_id": conversation_id, "user_id": user_id},
            )
            deleted_summary = bool(cur.rowcount)
        conn.commit()
    return DeleteConversationResponse(conversation_id=conversation_id, deleted_messages=deleted_messages, deleted_summary=deleted_summary)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, http_request: Request, identity: str = Depends(require_auth)) -> ChatResponse:
    check_rate_limit(http_request, identity)
    started_at = time.perf_counter()
    conversation_id = request.conversation_id or str(uuid.uuid4())
    try:
        serializable_payload = _retrieval_payload_for_message(request)
        # Use a single pooled connection for web search + conversation storage.
        with _get_pool().connection() as conn:
            _ensure_conversation_access(conn, conversation_id=conversation_id, user_id=request.user_id)
            conversation_summary = _load_conversation_summary(conn, conversation_id=conversation_id, user_id=request.user_id)
            recent_turns = _load_recent_conversation_turns(conn, conversation_id=conversation_id, user_id=request.user_id, limit=10)
            citations = _combined_citations(conn, serializable_payload["results"], request=request)
            references = _reference_cards(citations)
            chat_settings = with_model_override(load_chat_settings(), request.model)
            answer, chat_settings = generate_chat_answer(question=request.message, citations=citations, conversation_summary=conversation_summary, recent_turns=recent_turns, thinking_enabled=request.thinking_enabled, thinking_level=request.thinking_level, settings=chat_settings)
            follow_up_questions = _generate_follow_ups_safely(request=request, answer=answer, citations=citations, conversation_summary=conversation_summary, recent_turns=recent_turns, chat_settings=chat_settings)
            updated_summary = _summarize_conversation_safely(conn, conversation_id=conversation_id, user_id=request.user_id, previous_summary=conversation_summary, recent_turns=recent_turns, latest_user_message=request.message, latest_assistant_answer=answer, chat_settings=chat_settings)
            _store_conversation_turn(conn, conversation_id, "user", request.message, {"identity": identity, "user_id": request.user_id})
            _store_conversation_turn(conn, conversation_id, "assistant", answer, {"identity": identity, "user_id": request.user_id, "provider": chat_settings.provider, "model": chat_settings.model, "citations": citations, "references": references, "retrieval": serializable_payload, "follow_up_questions": follow_up_questions, "conversation_summary": updated_summary, "thinking_enabled": request.thinking_enabled, "thinking_level": request.thinking_level if request.thinking_enabled else None})
        response = ChatResponse(conversation_id=conversation_id, answer=answer, citations=citations, references=references, retrieval=SearchResponse(**serializable_payload), model=f"{chat_settings.provider}:{chat_settings.model}", follow_up_questions=follow_up_questions, conversation_summary=updated_summary, thinking_enabled=request.thinking_enabled, thinking_level=request.thinking_level if request.thinking_enabled else None)
        _record_rag_metric(
            endpoint="/chat",
            started_at=started_at,
            conversation_id=conversation_id,
            user_id=request.user_id,
            query=request.message,
            provider=chat_settings.provider,
            model=chat_settings.model,
            status="ok",
            retrieval_count=len(serializable_payload.get("results", [])),
            citation_count=len(citations),
            web_citation_count=sum(1 for citation in citations if citation.get("source_type") == "web_search"),
            answer_chars=len(answer),
            cache_type=(serializable_payload.get("cache") or {}).get("type"),
            payload={"identity": identity, "web_search": request.web_search, "thinking_enabled": request.thinking_enabled},
        )
        return response
    except HTTPException as exc:
        _record_rag_metric(endpoint="/chat", started_at=started_at, conversation_id=conversation_id, user_id=request.user_id, query=request.message, status="error", error=str(exc.detail), payload={"identity": identity})
        raise
    except Exception as exc:
        _record_rag_metric(endpoint="/chat", started_at=started_at, conversation_id=conversation_id, user_id=request.user_id, query=request.message, status="error", error=str(exc), payload={"identity": identity})
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat/stream")
def chat_stream(request: ChatRequest, http_request: Request, identity: str = Depends(require_auth)) -> StreamingResponse:
    check_rate_limit(http_request, identity)
    conversation_id = request.conversation_id or str(uuid.uuid4())

    def event_stream():
        started_at = time.perf_counter()
        try:
            serializable_payload = _retrieval_payload_for_message(request)
            with _get_pool().connection() as conn:
                _ensure_conversation_access(conn, conversation_id=conversation_id, user_id=request.user_id)
                conversation_summary = _load_conversation_summary(conn, conversation_id=conversation_id, user_id=request.user_id)
                recent_turns = _load_recent_conversation_turns(conn, conversation_id=conversation_id, user_id=request.user_id, limit=10)
                citations = _combined_citations(conn, serializable_payload["results"], request=request)
                references = _reference_cards(citations)
                yield _sse("retrieval", {"conversation_id": conversation_id, "cache": serializable_payload.get("cache", {}), "citations": citations, "references": references})
                chat_settings = with_model_override(load_chat_settings(), request.model)
                answer, chat_settings = generate_chat_answer(question=request.message, citations=citations, conversation_summary=conversation_summary, recent_turns=recent_turns, thinking_enabled=request.thinking_enabled, thinking_level=request.thinking_level, settings=chat_settings)
                follow_up_questions = _generate_follow_ups_safely(request=request, answer=answer, citations=citations, conversation_summary=conversation_summary, recent_turns=recent_turns, chat_settings=chat_settings)
                updated_summary = _summarize_conversation_safely(conn, conversation_id=conversation_id, user_id=request.user_id, previous_summary=conversation_summary, recent_turns=recent_turns, latest_user_message=request.message, latest_assistant_answer=answer, chat_settings=chat_settings)
                _store_conversation_turn(conn, conversation_id, "user", request.message, {"identity": identity, "user_id": request.user_id})
                _store_conversation_turn(conn, conversation_id, "assistant", answer, {"identity": identity, "user_id": request.user_id, "provider": chat_settings.provider, "model": chat_settings.model, "citations": citations, "references": references, "retrieval": serializable_payload, "follow_up_questions": follow_up_questions, "conversation_summary": updated_summary, "thinking_enabled": request.thinking_enabled, "thinking_level": request.thinking_level if request.thinking_enabled else None})
            for chunk in _chunk_text(answer):
                yield _sse("answer_chunk", {"text": chunk})
            _record_rag_metric(
                endpoint="/chat/stream",
                started_at=started_at,
                conversation_id=conversation_id,
                user_id=request.user_id,
                query=request.message,
                provider=chat_settings.provider,
                model=chat_settings.model,
                status="ok",
                retrieval_count=len(serializable_payload.get("results", [])),
                citation_count=len(citations),
                web_citation_count=sum(1 for citation in citations if citation.get("source_type") == "web_search"),
                answer_chars=len(answer),
                cache_type=(serializable_payload.get("cache") or {}).get("type"),
                payload={"identity": identity, "web_search": request.web_search, "thinking_enabled": request.thinking_enabled},
            )
            yield _sse("complete", {"conversation_id": conversation_id, "provider": chat_settings.provider, "model": chat_settings.model, "citations": citations, "references": references, "follow_up_questions": follow_up_questions, "conversation_summary": updated_summary, "thinking_enabled": request.thinking_enabled, "thinking_level": request.thinking_level if request.thinking_enabled else None})
        except Exception as exc:
            _record_rag_metric(endpoint="/chat/stream", started_at=started_at, conversation_id=conversation_id, user_id=request.user_id, query=request.message, status="error", error=str(exc), payload={"identity": identity})
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/conversation")
def conversation(request: ConversationRequest, http_request: Request, identity: str = Depends(require_auth)) -> StreamingResponse:
    check_rate_limit(http_request, identity)
    conversation_id = request.conversation_id or str(uuid.uuid4())

    def event_stream():
        started_at = time.perf_counter()
        try:
            payload = search_articles(
                database_url=_database_url(),
                embedding_endpoint_url=_embedding_url(),
                query=request.message,
                limit=request.limit,
                weights=request.weights.to_weights() if request.weights else None,
                rrf_k=request.rrf_k,
            )
            serializable_payload = _payload_to_response_dict(payload)
            citations = _citation_cards(serializable_payload["results"])
            answer = _extractive_answer(request.message, citations)
            with _get_pool().connection() as conn:
                _ensure_conversation_access(conn, conversation_id=conversation_id, user_id=request.user_id)
                _store_conversation_turn(conn, conversation_id, "user", request.message, {"identity": identity, "user_id": request.user_id})
                _store_conversation_turn(conn, conversation_id, "assistant", answer, {"identity": identity, "user_id": request.user_id, "citations": citations, "retrieval": serializable_payload})

            yield _sse("retrieval", {"conversation_id": conversation_id, "cache": serializable_payload.get("cache", {}), "citations": citations})
            for chunk in _chunk_text(answer):
                yield _sse("answer_chunk", {"text": chunk})
            _record_rag_metric(
                endpoint="/conversation",
                started_at=started_at,
                conversation_id=conversation_id,
                user_id=request.user_id,
                query=request.message,
                status="ok",
                retrieval_count=len(serializable_payload.get("results", [])),
                citation_count=len(citations),
                answer_chars=len(answer),
                cache_type=(serializable_payload.get("cache") or {}).get("type"),
                payload={"identity": identity},
            )
            yield _sse("complete", {"conversation_id": conversation_id, "citations": citations})
        except Exception as exc:
            _record_rag_metric(endpoint="/conversation", started_at=started_at, conversation_id=conversation_id, user_id=request.user_id, query=request.message, status="error", error=str(exc), payload={"identity": identity})
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")



def _record_rag_metric(
    *,
    endpoint: str,
    started_at: float,
    status: str,
    conversation_id: str | None = None,
    user_id: str = DEFAULT_USER_ID,
    query: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    retrieval_count: int = 0,
    citation_count: int = 0,
    web_citation_count: int = 0,
    answer_chars: int = 0,
    cache_type: str | None = None,
    error: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    latency_ms = max((time.perf_counter() - started_at) * 1000.0, 0.0)
    try:
        with _get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rag_request_metrics(
                        endpoint, conversation_id, user_id, query, provider, model, status,
                        latency_ms, retrieval_count, citation_count, web_citation_count,
                        answer_chars, cache_type, error, payload
                    )
                    VALUES (
                        %(endpoint)s, %(conversation_id)s, %(user_id)s, %(query)s, %(provider)s, %(model)s, %(status)s,
                        %(latency_ms)s, %(retrieval_count)s, %(citation_count)s, %(web_citation_count)s,
                        %(answer_chars)s, %(cache_type)s, %(error)s, %(payload)s::jsonb
                    );
                    """,
                    {
                        "endpoint": endpoint,
                        "conversation_id": conversation_id,
                        "user_id": user_id,
                        "query": query,
                        "provider": provider,
                        "model": model,
                        "status": status,
                        "latency_ms": latency_ms,
                        "retrieval_count": retrieval_count,
                        "citation_count": citation_count,
                        "web_citation_count": web_citation_count,
                        "answer_chars": answer_chars,
                        "cache_type": cache_type,
                        "error": error[:1000] if error else None,
                        "payload": json.dumps(to_jsonable(payload or {})),
                    },
                )
            conn.commit()
    except Exception:
        logger.exception("Failed to record RAG metric endpoint=%s status=%s", endpoint, status)


def _load_monitoring_summary(conn, *, window_hours: int) -> dict[str, Any]:
    params = {"window_hours": window_hours}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(*)::int AS total_requests,
                COUNT(*) FILTER (WHERE status <> 'ok')::int AS errors,
                COALESCE(AVG(latency_ms), 0)::float AS avg_latency_ms,
                COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms), 0)::float AS p95_latency_ms,
                COALESCE(AVG(citation_count), 0)::float AS avg_citations,
                COALESCE(AVG(retrieval_count), 0)::float AS avg_retrieval_count
            FROM rag_request_metrics
            WHERE created_at >= now() - (%(window_hours)s || ' hours')::interval;
            """,
            params,
        )
        summary = cur.fetchone() or {}
        cur.execute(
            """
            SELECT
                endpoint,
                COUNT(*)::int AS requests,
                COUNT(*) FILTER (WHERE status <> 'ok')::int AS errors,
                COALESCE(AVG(latency_ms), 0)::float AS avg_latency_ms,
                COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms), 0)::float AS p95_latency_ms,
                COALESCE(AVG(citation_count), 0)::float AS avg_citations
            FROM rag_request_metrics
            WHERE created_at >= now() - (%(window_hours)s || ' hours')::interval
            GROUP BY endpoint
            ORDER BY requests DESC, endpoint;
            """,
            params,
        )
        by_endpoint = [dict(row) for row in cur.fetchall()]
        cur.execute(
            """
            SELECT endpoint, conversation_id, user_id, query, error, created_at
            FROM rag_request_metrics
            WHERE created_at >= now() - (%(window_hours)s || ' hours')::interval
              AND status <> 'ok'
            ORDER BY created_at DESC
            LIMIT 10;
            """,
            params,
        )
        recent_errors = [dict(row) for row in cur.fetchall()]

    total = int(summary.get("total_requests") or 0)
    errors = int(summary.get("errors") or 0)
    return {
        "window_hours": window_hours,
        "total_requests": total,
        "errors": errors,
        "error_rate": (errors / total) if total else 0.0,
        "avg_latency_ms": float(summary.get("avg_latency_ms") or 0),
        "p95_latency_ms": float(summary.get("p95_latency_ms") or 0),
        "avg_citations": float(summary.get("avg_citations") or 0),
        "avg_retrieval_count": float(summary.get("avg_retrieval_count") or 0),
        "by_endpoint": by_endpoint,
        "recent_errors": recent_errors,
    }


def _ensure_conversation_access(conn, *, conversation_id: str, user_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(*) AS row_count,
                BOOL_OR(COALESCE(payload->>'user_id', %(default_user_id)s) = %(user_id)s) AS user_matches
            FROM rag_conversations
            WHERE conversation_id = %(conversation_id)s;
            """,
            {"conversation_id": conversation_id, "user_id": user_id, "default_user_id": DEFAULT_USER_ID},
        )
        row = cur.fetchone()
    if row and row["row_count"] and not row["user_matches"]:
        raise HTTPException(status_code=403, detail="conversation does not belong to user_id")


def _load_conversation_summary(conn, *, conversation_id: str, user_id: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT summary
            FROM rag_conversation_summaries
            WHERE conversation_id = %(conversation_id)s
              AND user_id = %(user_id)s;
            """,
            {"conversation_id": conversation_id, "user_id": user_id},
        )
        row = cur.fetchone()
    return row["summary"] if row else None


def _load_recent_conversation_turns(conn, *, conversation_id: str, user_id: str, limit: int = 10) -> list[dict[str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT role, content
            FROM rag_conversations
            WHERE conversation_id = %(conversation_id)s
              AND COALESCE(payload->>'user_id', %(default_user_id)s) = %(user_id)s
            ORDER BY created_at DESC
            LIMIT %(limit)s;
            """,
            {"conversation_id": conversation_id, "user_id": user_id, "default_user_id": DEFAULT_USER_ID, "limit": limit},
        )
        rows = cur.fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def _upsert_conversation_summary(conn, *, conversation_id: str, user_id: str, summary: str, turn_count: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO rag_conversation_summaries(conversation_id, user_id, summary, turn_count)
            VALUES (%(conversation_id)s, %(user_id)s, %(summary)s, %(turn_count)s)
            ON CONFLICT (conversation_id) DO UPDATE SET
                user_id = EXCLUDED.user_id,
                summary = EXCLUDED.summary,
                turn_count = EXCLUDED.turn_count,
                updated_at = now();
            """,
            {"conversation_id": conversation_id, "user_id": user_id, "summary": summary, "turn_count": turn_count},
        )
    conn.commit()


def _summarize_conversation_safely(conn, *, conversation_id: str, user_id: str, previous_summary: str | None, recent_turns: list[dict[str, str]], latest_user_message: str, latest_assistant_answer: str, chat_settings) -> str | None:
    try:
        summary, _ = generate_chat_summary(previous_summary=previous_summary, recent_turns=recent_turns, latest_user_message=latest_user_message, latest_assistant_answer=latest_assistant_answer, settings=chat_settings)
    except Exception:
        logger.exception("Failed to summarize conversation conversation_id=%s user_id=%s", conversation_id, user_id)
        return previous_summary
    if summary:
        _upsert_conversation_summary(conn, conversation_id=conversation_id, user_id=user_id, summary=summary, turn_count=len(recent_turns) + 2)
        return summary
    return previous_summary


def _generate_follow_ups_safely(*, request: ChatRequest, answer: str, citations: list[dict[str, Any]], conversation_summary: str | None, recent_turns: list[dict[str, str]], chat_settings) -> list[str]:
    try:
        follow_ups, _ = generate_follow_up_questions(question=request.message, answer=answer, citations=citations, conversation_summary=conversation_summary, recent_turns=recent_turns, settings=chat_settings)
    except Exception:
        logger.exception("Failed to generate follow-up questions conversation_id=%s user_id=%s", request.conversation_id, request.user_id)
        return []
    return follow_ups[:3]


def _retrieval_payload_for_message(request: ChatRequest | ConversationRequest) -> dict[str, Any]:
    payload = search_articles(
        database_url=_database_url(),
        embedding_endpoint_url=_embedding_url(),
        query=request.message,
        limit=request.limit,
        weights=request.weights.to_weights() if request.weights else None,
        rrf_k=request.rrf_k,
    )
    return _payload_to_response_dict(payload)


def _search_response_from_payload(payload: dict[str, Any]) -> SearchResponse:
    return SearchResponse(
        query=payload["query"],
        limit=payload["limit"],
        rrf_k=payload["rrf_k"],
        weights=payload["weights"],
        entity_candidates=payload["entity_candidates"],
        results=[
            SearchResultResponse(
                article_id=result.article_id if hasattr(result, "article_id") else result["article_id"],
                rrf_score=result.rrf_score if hasattr(result, "rrf_score") else result["rrf_score"],
                title=result.title if hasattr(result, "title") else result.get("title"),
                url=result.url if hasattr(result, "url") else result.get("url"),
                provider=result.provider if hasattr(result, "provider") else result.get("provider"),
                published_at=result.published_at if hasattr(result, "published_at") else result.get("published_at"),
                stream_scores=result.stream_scores if hasattr(result, "stream_scores") else result.get("stream_scores", {}),
                stream_ranks=result.stream_ranks if hasattr(result, "stream_ranks") else result.get("stream_ranks", {}),
                matches=[MatchResponse(**(asdict(match) if hasattr(match, "article_id") else match)) for match in (result.matches if hasattr(result, "matches") else result.get("matches", []))],
            )
            for result in payload["results"]
        ],
        cache=payload.get("cache", {}),
    )


def _payload_to_response_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return _search_response_from_payload(payload).model_dump(mode="json")


def _combined_citations(conn, results: list[dict[str, Any]], *, request: ChatRequest) -> list[dict[str, Any]]:
    """Build unified citations list, optionally augmented with live web search results.

    The caller is responsible for providing an open connection from the pool so
    this function does not open its own connection.
    """
    # Collect keywords for context engineering:
    # 1. Meaningful words from the user query (stop-words removed)
    # 2. Entity texts surfaced by the entity stream (e.g. "Micron", "Sarah Breeden")
    query_keywords: list[str] = extract_query_keywords(request.message)
    for result in results:
        for match in result.get("matches", []):
            entity_text = match.get("entity_text") if isinstance(match, dict) else getattr(match, "entity_text", None)
            if entity_text:
                query_keywords.append(entity_text)

    rag_citations = _citation_cards(results, start_index=1, query_keywords=query_keywords)
    if not request.web_search:
        return rag_citations
    web_citations = fetch_and_index_web_search(conn, query=request.message, limit=request.web_search_limit)
    return _renumber_citations([*rag_citations, *web_citations])


def _renumber_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    renumbered = []
    for index, citation in enumerate(citations, start=1):
        item = dict(citation)
        item["citation_index"] = index
        item["citation_marker"] = f"[{index}]"
        renumbered.append(item)
    return renumbered


def _citation_cards(
    results: list[dict[str, Any]],
    *,
    start_index: int = 1,
    query_keywords: list[str] | None = None,
) -> list[dict[str, Any]]:
    cards = []
    for index, result in enumerate(results, start=start_index):
        best_match = _best_snippet_match(
            result.get("matches") or [],
            query_keywords=query_keywords or [],
        )
        cards.append(
            {
                "source_type": "rag_article",
                "citation_index": index,
                "citation_marker": f"[{index}]",
                "article_id": result["article_id"],
                "web_search_id": None,
                "title": result.get("title"),
                "url": result.get("url"),
                "provider": result.get("provider"),
                "published_at": result.get("published_at"),
                "rrf_score": result.get("rrf_score"),
                "stream_ranks": result.get("stream_ranks", {}),
                "snippet": best_match.get("snippet"),
                "char_start": best_match.get("char_start"),
                "char_end": best_match.get("char_end"),
            }
        )
    return cards


def _best_snippet_match(
    matches: list[dict[str, Any]],
    *,
    query_keywords: list[str] | None = None,
) -> dict[str, Any]:
    """Pick the most informative snippet from an article's stream matches.

    Priority:
    1. vector    — full 500-token chunk, highest semantic density
    2. full_text — ts_headline() contextual window, good for exact-term queries
    3. entity    — full ML document text, keyword-context extracted before use
    4. any other match with a snippet
    5. first match regardless (fallback)

    For entity stream matches the snippet is the entire article ML document.
    We apply extract_keyword_context to reduce it to only the sentences relevant
    to the user query / entity, expanding at most 5 gap sentences in each
    direction from each anchor sentence.
    """
    stream_priority = {"vector": 0, "full_text": 1, "entity": 2}
    candidates = [m for m in matches if m.get("snippet")]
    if not candidates:
        return matches[0] if matches else {}

    best = min(candidates, key=lambda m: stream_priority.get(m.get("stream", ""), 99))

    # Apply context engineering only to entity stream snippets (full doc text).
    # Vector and FTS snippets are already focused windows — leave them unchanged.
    if best.get("stream") == "entity" and query_keywords:
        raw_snippet = best.get("snippet") or ""
        engineered = extract_keyword_context(raw_snippet, query_keywords, window=5)
        if engineered != raw_snippet:
            # Return a shallow copy so we don't mutate the original match dict.
            best = {**best, "snippet": engineered}

    return best



def _reference_cards(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source_type": citation.get("source_type", "rag_article"),
            "citation_index": citation.get("citation_index"),
            "citation_marker": citation.get("citation_marker"),
            "article_id": citation.get("article_id"),
            "web_search_id": citation.get("web_search_id"),
            "title": citation.get("title"),
            "url": citation.get("url"),
            "provider": citation.get("provider"),
            "published_at": citation.get("published_at"),
            "snippet": citation.get("snippet"),
            "char_start": citation.get("char_start"),
            "char_end": citation.get("char_end"),
            "rrf_score": citation.get("rrf_score"),
            "stream_ranks": citation.get("stream_ranks", {}),
        }
        for citation in citations
    ]


def _extractive_answer(message: str, citations: list[dict[str, Any]]) -> str:
    if not citations:
        return f"I could not find indexed article evidence for: {message}"
    top_lines = []
    for index, citation in enumerate(citations[:5], start=1):
        title = citation.get("title") or "Untitled article"
        provider = citation.get("provider") or "unknown source"
        snippet = citation.get("snippet") or "No snippet available."
        top_lines.append(f"{index}. {title} ({provider}): {snippet}")
    return "I found relevant article evidence. Top retrieved sources:\n" + "\n".join(top_lines)


def _chunk_text(text: str, *, size: int = 240):
    for start in range(0, len(text), size):
        yield text[start : start + size]


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _store_conversation_turn(conn, conversation_id: str, role: str, content: str, payload: dict[str, Any]) -> None:
    """Persist a single conversation turn using the provided connection.

    The caller controls the connection lifetime; no new connection is opened here.
    Schema must already be initialised (guaranteed by the lifespan hook).
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO rag_conversations(conversation_id, role, content, payload)
            VALUES (%(conversation_id)s, %(role)s, %(content)s, %(payload)s::jsonb);
            """,
            {"conversation_id": conversation_id, "role": role, "content": content, "payload": json.dumps(to_jsonable(payload))},
        )
    conn.commit()
