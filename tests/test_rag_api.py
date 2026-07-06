import json

import pytest
from fastapi import HTTPException

from news_ingest.ml.rag_api import _citation_cards, _ensure_conversation_access, _reference_cards, _sse


def test_sse_formats_named_event_with_json_payload() -> None:
    event = _sse("answer_chunk", {"text": "hello"})

    assert event.startswith("event: answer_chunk\n")
    assert json.loads(event.split("data: ", 1)[1]) == {"text": "hello"}


def test_citation_cards_use_first_snippet_match() -> None:
    cards = _citation_cards([{"article_id": 10, "title": "A", "url": "u", "provider": "p", "published_at": None, "rrf_score": 0.1, "stream_ranks": {"vector": 1}, "matches": [{"snippet": None}, {"snippet": "matched", "char_start": 5, "char_end": 12}]}])

    assert cards == [{"source_type": "rag_article", "citation_index": 1, "citation_marker": "[1]", "article_id": 10, "web_search_id": None, "title": "A", "url": "u", "provider": "p", "published_at": None, "rrf_score": 0.1, "stream_ranks": {"vector": 1}, "snippet": "matched", "char_start": 5, "char_end": 12}]


def test_reference_cards_map_markers_to_article_metadata() -> None:
    references = _reference_cards([{"source_type": "web_search", "citation_index": 1, "citation_marker": "[1]", "article_id": None, "web_search_id": 10, "title": "A", "url": "u", "provider": "web:duckduckgo", "published_at": "date", "snippet": "s", "char_start": 1, "char_end": 2, "rrf_score": 0.2, "stream_ranks": {"web_search": 1}}])

    assert references == [{"source_type": "web_search", "citation_index": 1, "citation_marker": "[1]", "article_id": None, "web_search_id": 10, "title": "A", "url": "u", "provider": "web:duckduckgo", "published_at": "date", "snippet": "s", "char_start": 1, "char_end": 2, "rrf_score": 0.2, "stream_ranks": {"web_search": 1}}]


class _Cursor:
    def __init__(self, row):
        self.row = row

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params):
        self.query = query
        self.params = params

    def fetchone(self):
        return self.row


class _Conn:
    def __init__(self, row):
        self.row = row

    def cursor(self):
        return _Cursor(self.row)


def test_ensure_conversation_access_allows_empty_or_matching_conversation() -> None:
    _ensure_conversation_access(_Conn({"row_count": 0, "user_matches": None}), conversation_id="c1", user_id="user_id")
    _ensure_conversation_access(_Conn({"row_count": 2, "user_matches": True}), conversation_id="c1", user_id="user_id")


def test_ensure_conversation_access_rejects_different_user_id() -> None:
    with pytest.raises(HTTPException) as exc_info:
        _ensure_conversation_access(_Conn({"row_count": 2, "user_matches": False}), conversation_id="c1", user_id="other")

    assert exc_info.value.status_code == 403
