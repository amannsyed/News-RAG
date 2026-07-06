from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Stop-words used by extract_query_keywords
# ---------------------------------------------------------------------------
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "of", "off", "over", "under", "then",
    "and", "but", "or", "nor", "yet", "so", "if", "as", "that", "what",
    "which", "who", "whom", "this", "these", "those", "it", "its",
    "not", "no", "how", "when", "where", "why", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such",
    "than", "too", "very", "just", "also", "here", "there", "their",
    "they", "them", "we", "our", "you", "your", "he", "she", "his", "her",
})


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    char_start: int
    char_end: int
    token_count: int
    content_hash: str


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def stable_hash(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def build_article_document(*, title: str | None, description: str | None, content: str | None) -> str:
    parts = []
    for value in (title, description, content):
        normalized = normalize_text(value)
        if normalized and normalized not in parts:
            parts.append(normalized)
    return "\n\n".join(parts)


def chunk_text(text: str, *, max_tokens: int = 500, overlap_tokens: int = 50) -> list[TextChunk]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be non-negative and smaller than max_tokens")

    tokens = [(match.group(0), match.start(), match.end()) for match in re.finditer(r"\S+", text)]
    if not tokens:
        return []

    chunks: list[TextChunk] = []
    start_token = 0
    step = max_tokens - overlap_tokens
    while start_token < len(tokens):
        end_token = min(start_token + max_tokens, len(tokens))
        char_start = tokens[start_token][1]
        char_end = tokens[end_token - 1][2]
        chunk = text[char_start:char_end]
        chunks.append(
            TextChunk(
                index=len(chunks),
                text=chunk,
                char_start=char_start,
                char_end=char_end,
                token_count=end_token - start_token,
                content_hash=stable_hash(chunk),
            )
        )
        if end_token >= len(tokens):
            break
        start_token += step
    return chunks


def chunks_from_token_offsets(text: str, offsets: list[tuple[int, int]], *, max_tokens: int = 500, overlap_tokens: int = 50) -> list[TextChunk]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be non-negative and smaller than max_tokens")

    clean_offsets = [(int(start), int(end)) for start, end in offsets if int(end) > int(start)]
    if not clean_offsets:
        return []

    chunks: list[TextChunk] = []
    start_token = 0
    step = max_tokens - overlap_tokens
    while start_token < len(clean_offsets):
        end_token = min(start_token + max_tokens, len(clean_offsets))
        char_start = clean_offsets[start_token][0]
        char_end = clean_offsets[end_token - 1][1]
        chunk = text[char_start:char_end]
        chunks.append(
            TextChunk(
                index=len(chunks),
                text=chunk,
                char_start=char_start,
                char_end=char_end,
                token_count=end_token - start_token,
                content_hash=stable_hash(chunk),
            )
        )
        if end_token >= len(clean_offsets):
            break
        start_token += step
    return chunks


# ---------------------------------------------------------------------------
# Context engineering helpers
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Split text into individual sentences.

    Handles standard punctuation boundaries (. ! ?) and paragraph breaks
    (double or single newlines common in structured news feeds).
    """
    raw = re.split(r'(?<=[.!?])\s+|\n{2,}|\n', text.strip())
    return [s.strip() for s in raw if s.strip()]


def extract_query_keywords(query: str) -> list[str]:
    """Return meaningful content words from a query string.

    Strips stop-words, lowercases for deduplication, and preserves insertion
    order. Words must be at least 3 characters so single-letter tokens are
    excluded, but short acronyms like "AI" or "EU" still pass.
    """
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]{2,}\b', query)
    seen: set[str] = set()
    result: list[str] = []
    for w in words:
        lower = w.lower()
        if lower not in _STOP_WORDS and lower not in seen:
            seen.add(lower)
            result.append(w)
    return result


def extract_keyword_context(
    text: str,
    keywords: list[str],
    *,
    window: int = 5,
) -> str:
    """Extract contextually relevant sentences from *text* anchored on *keywords*.

    Algorithm
    ---------
    1. Split the text into sentences.
    2. Mark every sentence that contains at least one keyword as an **anchor**.
    3. From each anchor expand outward sentence by sentence in both directions:
       - If the next sentence contains a keyword, reset the gap counter and continue.
       - If it does not, include it but increment a gap counter.
       - Stop in that direction once *window* consecutive non-keyword sentences
         have been accumulated.
    4. Union the collected indices (overlapping windows merge automatically).
    5. Reassemble in document order, inserting "..." between non-consecutive
       ranges so the reader can see where text was omitted.

    Edge cases
    ----------
    - **First sentence**: backward expansion is clamped at index 0 — no underflow.
    - **Last sentence**: forward expansion is clamped at the final index.
    - **No keyword found**: the original *text* is returned unchanged so callers
      always receive a non-empty string.
    - **Empty keyword list**: the original *text* is returned unchanged.
    """
    sentences = split_sentences(text)
    if not sentences:
        return text

    kws_lower = [k.lower() for k in keywords if k.strip()]
    if not kws_lower:
        return text

    def _has_keyword(sentence: str) -> bool:
        lower_s = sentence.lower()
        return any(kw in lower_s for kw in kws_lower)

    n = len(sentences)
    anchors = [i for i, s in enumerate(sentences) if _has_keyword(s)]
    if not anchors:
        # None of the keywords appear in this document — return the whole text.
        return text

    included: set[int] = set()

    for anchor in anchors:
        included.add(anchor)

        # Expand backward from this anchor
        consecutive_gap = 0
        for i in range(anchor - 1, -1, -1):
            included.add(i)
            if _has_keyword(sentences[i]):
                consecutive_gap = 0          # keyword resets the gap counter
            else:
                consecutive_gap += 1
                if consecutive_gap >= window:  # stop: `window` gap sentences hit
                    break

        # Expand forward from this anchor
        consecutive_gap = 0
        for i in range(anchor + 1, n):
            included.add(i)
            if _has_keyword(sentences[i]):
                consecutive_gap = 0
            else:
                consecutive_gap += 1
                if consecutive_gap >= window:
                    break

    # Reassemble with "..." markers between non-consecutive sentence ranges
    sorted_indices = sorted(included)
    parts: list[str] = []
    prev_idx: int | None = None
    for idx in sorted_indices:
        if prev_idx is not None and idx > prev_idx + 1:
            parts.append("...")
        parts.append(sentences[idx])
        prev_idx = idx

    return " ".join(parts)
