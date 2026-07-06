import json
from urllib.error import HTTPError

import pytest

from news_ingest.ml.embedding_worker import vector_literal
from news_ingest.ml.http_client import ModelEndpointError, post_json


def test_vector_literal_requires_768_dimensions() -> None:
    with pytest.raises(ValueError):
        vector_literal([0.1, 0.2])


def test_vector_literal_formats_pgvector_value() -> None:
    literal = vector_literal([1.0] * 768)
    assert literal.startswith("[")
    assert literal.endswith("]")
    assert literal.count(",") == 767


def test_post_json_sends_payload_and_parses_response(monkeypatch) -> None:
    calls = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"ok": True}).encode("utf-8")

    def fake_urlopen(request, timeout):
        calls["url"] = request.full_url
        calls["body"] = json.loads(request.data.decode("utf-8"))
        calls["timeout"] = timeout
        return Response()

    monkeypatch.setattr("news_ingest.ml.http_client.urlopen", fake_urlopen)

    assert post_json("http://service/embed", {"texts": ["x"]}, timeout=5) == {"ok": True}
    assert calls == {"url": "http://service/embed", "body": {"texts": ["x"]}, "timeout": 5}


def test_post_json_wraps_http_errors(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 429, "Too Many", {}, None)

    monkeypatch.setattr("news_ingest.ml.http_client.urlopen", fake_urlopen)

    with pytest.raises(ModelEndpointError):
        post_json("http://service/embed", {"texts": ["x"]})

from news_ingest.ml.chunk_client import fetch_token_chunks


def test_fetch_token_chunks_calls_embedding_service_chunk_endpoint(monkeypatch) -> None:
    calls = {}

    def fake_post_json(url, payload, timeout=120):
        calls["url"] = url
        calls["payload"] = payload
        calls["timeout"] = timeout
        return {
            "chunks": [
                [
                    {
                        "index": 0,
                        "text": "hello",
                        "char_start": 0,
                        "char_end": 5,
                        "token_count": 1,
                        "content_hash": "abc",
                    }
                ]
            ]
        }

    monkeypatch.setattr("news_ingest.ml.chunk_client.post_json", fake_post_json)

    chunks = fetch_token_chunks("http://embedding", "hello", max_tokens=500, overlap_tokens=50)

    assert calls["url"] == "http://embedding/chunk"
    assert calls["payload"] == {"texts": ["hello"], "max_tokens": 500, "overlap_tokens": 50}
    assert chunks[0].text == "hello"
    assert chunks[0].token_count == 1
