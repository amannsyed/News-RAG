from __future__ import annotations

from typing import Any

from news_ingest.ml.http_client import post_json
from news_ingest.ml.text import TextChunk


def fetch_token_chunks(endpoint_url: str, text: str, *, max_tokens: int = 500, overlap_tokens: int = 50) -> list[TextChunk]:
    if not text:
        return []
    payload = post_json(
        endpoint_url.rstrip("/") + "/chunk",
        {"texts": [text], "max_tokens": max_tokens, "overlap_tokens": overlap_tokens},
        timeout=120,
    )
    chunk_sets = payload.get("chunks")
    if not isinstance(chunk_sets, list) or len(chunk_sets) != 1:
        raise RuntimeError("chunk endpoint response missing one chunk set")
    return [_chunk_from_payload(item) for item in chunk_sets[0]]


def _chunk_from_payload(item: dict[str, Any]) -> TextChunk:
    return TextChunk(
        index=int(item["index"]),
        text=str(item["text"]),
        char_start=int(item["char_start"]),
        char_end=int(item["char_end"]),
        token_count=int(item["token_count"]),
        content_hash=str(item["content_hash"]),
    )
