import pytest
from fastapi import HTTPException

from news_ingest.ml import rag_security


class Client:
    host = "127.0.0.1"


class Request:
    client = Client()


def test_require_auth_raises_503_when_token_unset(monkeypatch) -> None:
    monkeypatch.delenv("RAG_API_TOKEN", raising=False)
    monkeypatch.delenv("RAG_ALLOW_ANONYMOUS", raising=False)

    with pytest.raises(HTTPException) as exc:
        rag_security.require_auth()

    assert exc.value.status_code == 503


def test_require_auth_allows_anonymous_when_explicitly_opted_in(monkeypatch) -> None:
    monkeypatch.delenv("RAG_API_TOKEN", raising=False)
    monkeypatch.setenv("RAG_ALLOW_ANONYMOUS", "true")

    assert rag_security.require_auth() == "anonymous"


def test_require_auth_rejects_missing_bearer_token(monkeypatch) -> None:
    monkeypatch.setenv("RAG_API_TOKEN", "secret")

    with pytest.raises(HTTPException) as exc:
        rag_security.require_auth()

    assert exc.value.status_code == 401


def test_require_auth_accepts_matching_bearer_token(monkeypatch) -> None:
    monkeypatch.setenv("RAG_API_TOKEN", "secret")

    assert rag_security.require_auth("Bearer secret") == "token-user"


def test_rate_limit_uses_token_bucket(monkeypatch) -> None:
    rag_security._buckets.clear()
    monkeypatch.setenv("RAG_RATE_LIMIT_PER_MINUTE", "1")
    monkeypatch.setenv("RAG_RATE_LIMIT_BURST", "1")

    rag_security.check_rate_limit(Request(), "user-1")
    with pytest.raises(HTTPException) as exc:
        rag_security.check_rate_limit(Request(), "user-1")

    assert exc.value.status_code == 429
