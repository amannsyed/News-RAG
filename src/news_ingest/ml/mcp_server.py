from __future__ import annotations

import argparse
import json
import os
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from fastmcp import FastMCP


DEFAULT_RAG_API_URL = "http://localhost:8003"
DEFAULT_USER_ID = "user_id"
DEFAULT_CHAT_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
ChatModel = Literal[
    "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-sonnet-5",
    "anthropic.claude-opus-4-8",
]
ThinkingLevel = Literal["low", "medium", "max"]


def _rag_api_url() -> str:
    return os.getenv("MCP_RAG_API_URL", os.getenv("RAG_API_URL", DEFAULT_RAG_API_URL)).rstrip("/")


def _rag_api_token() -> str:
    return os.getenv("MCP_RAG_API_TOKEN", os.getenv("RAG_API_TOKEN", "")).strip()


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    token = _rag_api_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if extra:
        headers.update(extra)
    return headers


def _request_json(method: str, path: str, *, body: dict[str, Any] | None = None, query: dict[str, Any] | None = None, timeout: int = 180) -> dict[str, Any] | list[Any]:
    url = _build_url(path, query=query)
    data = None
    headers = _headers()
    if body is not None:
        data = json.dumps(_clean_payload(body)).encode("utf-8")
        headers = _headers({"Content-Type": "application/json"})
    request = Request(url, data=data, method=method.upper(), headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            if not payload:
                return {}
            return json.loads(payload)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method.upper()} {url} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"{method.upper()} {url} failed: {exc}") from exc


def _request_sse(path: str, *, body: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
    url = _build_url(path)
    request = Request(
        url,
        data=json.dumps(_clean_payload(body)).encode("utf-8"),
        method="POST",
        headers=_headers({"Content-Type": "application/json", "Accept": "text/event-stream"}),
    )
    events: list[dict[str, Any]] = []
    answer_chunks: list[str] = []
    try:
        with urlopen(request, timeout=timeout) as response:
            event_name = "message"
            data_lines: list[str] = []
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    _append_sse_event(events, answer_chunks, event_name, data_lines)
                    event_name = "message"
                    data_lines = []
                    continue
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data_lines.append(line.split(":", 1)[1].strip())
            _append_sse_event(events, answer_chunks, event_name, data_lines)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"POST {url} failed: {exc}") from exc

    complete = next((event["data"] for event in reversed(events) if event.get("event") == "complete"), {})
    errors = [event["data"] for event in events if event.get("event") == "error"]
    return {
        "answer": "".join(answer_chunks),
        "complete": complete,
        "errors": errors,
        "events": events,
    }


def _append_sse_event(events: list[dict[str, Any]], answer_chunks: list[str], event_name: str, data_lines: list[str]) -> None:
    if not data_lines:
        return
    raw_data = "\n".join(data_lines)
    try:
        data: Any = json.loads(raw_data)
    except json.JSONDecodeError:
        data = raw_data
    events.append({"event": event_name, "data": data})
    if event_name == "answer_chunk" and isinstance(data, dict):
        answer_chunks.append(str(data.get("text", "")))


def _build_url(path: str, *, query: dict[str, Any] | None = None) -> str:
    url = _rag_api_url() + path
    clean_query = _clean_payload(query or {})
    if clean_query:
        url += "?" + urlencode(clean_query, doseq=True)
    return url


def _clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


mcp = FastMCP(
    "News RAG MCP",
    instructions=(
        "Standalone MCP wrapper for the local News RAG FastAPI service. "
        "Use search_news for retrieval, chat_news for citation-grounded answers, "
        "monitoring_summary/evaluate_recent_chats for quick observability, and conversation tools to inspect or delete chat history."
    ),
)


@mcp.tool
def rag_health() -> dict[str, Any]:
    """Check whether the backing News RAG API is reachable."""
    return _request_json("GET", "/health")  # type: ignore[return-value]


@mcp.tool
def search_news(
    query: str,
    limit: int = 20,
    rrf_k: int = 60,
    vector_weight: float | None = None,
    full_text_weight: float | None = None,
    entity_weight: float | None = None,
) -> dict[str, Any]:
    """Run hybrid RAG retrieval over indexed news using vector, full-text, and entity search."""
    weights = None
    if any(value is not None for value in (vector_weight, full_text_weight, entity_weight)):
        weights = {
            "vector": vector_weight if vector_weight is not None else 0.6,
            "full_text": full_text_weight if full_text_weight is not None else 0.2,
            "entity": entity_weight if entity_weight is not None else 0.2,
        }
    return _request_json("POST", "/search", body={"query": query, "limit": limit, "rrf_k": rrf_k, "weights": weights})  # type: ignore[return-value]


@mcp.tool
def chat_news(
    message: str,
    conversation_id: str | None = None,
    user_id: str = DEFAULT_USER_ID,
    limit: int = 20,
    web_search: bool = False,
    web_search_limit: int = 20,
    model: ChatModel = DEFAULT_CHAT_MODEL,
    thinking_enabled: bool = False,
    thinking_level: ThinkingLevel = "medium",
    rrf_k: int = 60,
    vector_weight: float | None = None,
    full_text_weight: float | None = None,
    entity_weight: float | None = None,
) -> dict[str, Any]:
    """Ask the Claude-backed RAG chat endpoint and return the answer with citations and follow-ups."""
    return _request_json("POST", "/chat", body=_chat_payload(message=message, conversation_id=conversation_id, user_id=user_id, limit=limit, web_search=web_search, web_search_limit=web_search_limit, model=model, thinking_enabled=thinking_enabled, thinking_level=thinking_level, rrf_k=rrf_k, vector_weight=vector_weight, full_text_weight=full_text_weight, entity_weight=entity_weight))  # type: ignore[return-value]


@mcp.tool
def chat_news_stream(
    message: str,
    conversation_id: str | None = None,
    user_id: str = DEFAULT_USER_ID,
    limit: int = 20,
    web_search: bool = False,
    web_search_limit: int = 20,
    model: ChatModel = DEFAULT_CHAT_MODEL,
    thinking_enabled: bool = False,
    thinking_level: ThinkingLevel = "medium",
    rrf_k: int = 60,
) -> dict[str, Any]:
    """Call /chat/stream, collect SSE events, and return the final streamed answer."""
    return _request_sse("/chat/stream", body=_chat_payload(message=message, conversation_id=conversation_id, user_id=user_id, limit=limit, web_search=web_search, web_search_limit=web_search_limit, model=model, thinking_enabled=thinking_enabled, thinking_level=thinking_level, rrf_k=rrf_k, vector_weight=None, full_text_weight=None, entity_weight=None))


@mcp.tool
def extractive_conversation_stream(
    message: str,
    conversation_id: str | None = None,
    user_id: str = DEFAULT_USER_ID,
    limit: int = 20,
    rrf_k: int = 60,
) -> dict[str, Any]:
    """Call the extractive /conversation SSE endpoint and return collected events and answer text."""
    return _request_sse("/conversation", body={"message": message, "conversation_id": conversation_id, "user_id": user_id, "limit": limit, "rrf_k": rrf_k})


@mcp.tool
def monitoring_summary(window_hours: int = 24) -> dict[str, Any]:
    """Return recent RAG API monitoring metrics: volume, latency, errors, citations, and endpoint breakdown."""
    return _request_json("GET", "/monitoring/summary", query={"window_hours": window_hours})  # type: ignore[return-value]


@mcp.tool
def evaluate_recent_chats(limit: int = 25, user_id: str | None = None) -> dict[str, Any]:
    """Run lightweight deterministic evaluation over recent stored chat answers."""
    return _request_json("GET", "/evaluation/recent", query={"limit": limit, "user_id": user_id})  # type: ignore[return-value]


@mcp.tool
def list_conversations(user_id: str = DEFAULT_USER_ID) -> list[Any]:
    """List stored chat conversations for a user_id."""
    result = _request_json("GET", "/conversations", query={"user_id": user_id})
    return result if isinstance(result, list) else [result]


@mcp.tool
def get_conversation(conversation_id: str, user_id: str = DEFAULT_USER_ID) -> list[Any]:
    """Fetch stored messages for a conversation."""
    result = _request_json("GET", f"/conversations/{conversation_id}", query={"user_id": user_id})
    return result if isinstance(result, list) else [result]


@mcp.tool
def delete_conversation(conversation_id: str, user_id: str = DEFAULT_USER_ID) -> dict[str, Any]:
    """Delete a stored conversation and its rolling summary for a user_id."""
    return _request_json("DELETE", f"/conversations/{conversation_id}", query={"user_id": user_id})  # type: ignore[return-value]


def _chat_payload(
    *,
    message: str,
    conversation_id: str | None,
    user_id: str,
    limit: int,
    web_search: bool,
    web_search_limit: int,
    model: str,
    thinking_enabled: bool,
    thinking_level: str,
    rrf_k: int,
    vector_weight: float | None,
    full_text_weight: float | None,
    entity_weight: float | None,
) -> dict[str, Any]:
    weights = None
    if any(value is not None for value in (vector_weight, full_text_weight, entity_weight)):
        weights = {
            "vector": vector_weight if vector_weight is not None else 0.6,
            "full_text": full_text_weight if full_text_weight is not None else 0.2,
            "entity": entity_weight if entity_weight is not None else 0.2,
        }
    return {
        "message": message,
        "conversation_id": conversation_id,
        "user_id": user_id,
        "limit": limit,
        "rrf_k": rrf_k,
        "weights": weights,
        "web_search": web_search,
        "web_search_limit": web_search_limit,
        "model": model,
        "thinking_enabled": thinking_enabled,
        "thinking_level": thinking_level,
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the News RAG FastMCP server.")
    parser.add_argument("--transport", choices=("stdio", "http", "sse", "streamable-http"), default=os.getenv("MCP_TRANSPORT", "stdio"))
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "8004")))
    parser.add_argument("--path", default=os.getenv("MCP_PATH", "/mcp"))
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port, path=args.path)


if __name__ == "__main__":
    main()
