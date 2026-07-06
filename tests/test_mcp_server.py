from news_ingest.ml import mcp_server


def test_clean_payload_removes_none_but_keeps_false_and_zero() -> None:
    assert mcp_server._clean_payload({"a": None, "b": False, "c": 0, "d": ""}) == {"b": False, "c": 0, "d": ""}


def test_build_url_uses_mcp_rag_api_url(monkeypatch) -> None:
    monkeypatch.setenv("MCP_RAG_API_URL", "http://rag-api:8003/")

    assert mcp_server._build_url("/conversations", query={"user_id": "user_id"}) == "http://rag-api:8003/conversations?user_id=user_id"


def test_headers_prefers_mcp_token(monkeypatch) -> None:
    monkeypatch.setenv("RAG_API_TOKEN", "rag-token")
    monkeypatch.setenv("MCP_RAG_API_TOKEN", "mcp-token")

    assert mcp_server._headers()["Authorization"] == "Bearer mcp-token"


def test_append_sse_event_collects_answer_chunk() -> None:
    events = []
    chunks = []

    mcp_server._append_sse_event(events, chunks, "answer_chunk", ['{"text":"hello"}'])

    assert chunks == ["hello"]
    assert events == [{"event": "answer_chunk", "data": {"text": "hello"}}]
