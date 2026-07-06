import pytest

from news_ingest.ml.text import (
    build_article_document,
    chunk_text,
    extract_keyword_context,
    extract_query_keywords,
    split_sentences,
    stable_hash,
)


def test_build_article_document_deduplicates_and_separates_parts() -> None:
    document = build_article_document(title=" Title ", description="Description", content="Title")
    assert document == "Title\n\nDescription"


def test_chunk_text_preserves_offsets_and_overlap() -> None:
    text = "one two three four five six"
    chunks = chunk_text(text, max_tokens=3, overlap_tokens=1)

    assert [chunk.text for chunk in chunks] == ["one two three", "three four five", "five six"]
    assert text[chunks[1].char_start : chunks[1].char_end] == "three four five"
    assert chunks[0].content_hash == stable_hash("one two three")


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("hello world", max_tokens=10, overlap_tokens=10)

from news_ingest.ml.text import chunks_from_token_offsets


def test_chunks_from_token_offsets_uses_model_offsets() -> None:
    text = "ab cd ef gh"
    offsets = [(0, 2), (3, 5), (6, 8), (9, 11)]

    chunks = chunks_from_token_offsets(text, offsets, max_tokens=3, overlap_tokens=1)

    assert [chunk.text for chunk in chunks] == ["ab cd ef", "ef gh"]
    assert chunks[0].token_count == 3
    assert chunks[1].char_start == 6


# ---------------------------------------------------------------------------
# split_sentences
# ---------------------------------------------------------------------------

def test_split_sentences_on_punctuation() -> None:
    text = "Micron beat estimates. Revenue rose sharply! What next?"
    assert split_sentences(text) == [
        "Micron beat estimates.",
        "Revenue rose sharply!",
        "What next?",
    ]


def test_split_sentences_on_newlines() -> None:
    text = "First paragraph.\n\nSecond paragraph.\nThird line."
    sentences = split_sentences(text)
    assert "First paragraph." in sentences
    assert "Second paragraph." in sentences


def test_split_sentences_strips_empty() -> None:
    assert split_sentences("  \n  \n  ") == []


# ---------------------------------------------------------------------------
# extract_query_keywords
# ---------------------------------------------------------------------------

def test_extract_query_keywords_removes_stopwords() -> None:
    keywords = extract_query_keywords("What is happening with Micron stock and earnings?")
    assert "Micron" in keywords
    assert "stock" in keywords
    assert "earnings" in keywords
    # stop-words must be excluded
    assert "what" not in keywords
    assert "is" not in keywords
    assert "with" not in keywords
    assert "and" not in keywords


def test_extract_query_keywords_deduplicates() -> None:
    keywords = extract_query_keywords("Micron Micron earnings earnings")
    assert keywords.count("Micron") == 1
    assert keywords.count("earnings") == 1


def test_extract_query_keywords_empty() -> None:
    assert extract_query_keywords("") == []


# ---------------------------------------------------------------------------
# extract_keyword_context
# ---------------------------------------------------------------------------

def test_extract_keyword_context_returns_anchor_and_neighbours() -> None:
    text = (
        "Sentence one has nothing. "
        "Sentence two has nothing. "
        "Sentence three mentions Micron. "
        "Sentence four has nothing. "
        "Sentence five has nothing."
    )
    result = extract_keyword_context(text, ["Micron"], window=5)
    assert "Micron" in result
    assert "three mentions Micron" in result


def test_extract_keyword_context_stops_after_window_gap_sentences() -> None:
    # 6 non-keyword sentences after the anchor — only first 5 should be included
    sentences = ["Keyword is here."] + [f"Gap sentence {i}." for i in range(1, 8)]
    text = " ".join(sentences)
    result = extract_keyword_context(text, ["Keyword"], window=5)
    # Gap sentence 6 and 7 (indices 6,7) should NOT appear — beyond the window
    assert "Gap sentence 6" not in result
    assert "Gap sentence 7" not in result
    # Gap sentences 1–5 should all be present
    for i in range(1, 6):
        assert f"Gap sentence {i}" in result


def test_extract_keyword_context_keyword_in_first_sentence() -> None:
    text = "Micron beats estimates. Revenue up 400 percent. Analysts impressed."
    result = extract_keyword_context(text, ["Micron"], window=5)
    # All sentences should be included (keyword at start, window covers the rest)
    assert "Revenue up 400 percent" in result
    assert "Analysts impressed" in result


def test_extract_keyword_context_keyword_in_last_sentence() -> None:
    text = "Revenue was strong. Demand kept rising. Micron delivered results."
    result = extract_keyword_context(text, ["Micron"], window=5)
    assert "Revenue was strong" in result
    assert "Micron delivered results" in result


def test_extract_keyword_context_no_keyword_returns_full_text() -> None:
    text = "Nothing relevant here. Just filler content."
    result = extract_keyword_context(text, ["Micron"], window=5)
    assert result == text


def test_extract_keyword_context_empty_keywords_returns_full_text() -> None:
    text = "Some article content."
    assert extract_keyword_context(text, []) == text


def test_extract_keyword_context_inserts_ellipsis_between_gaps() -> None:
    # Two keyword anchors far apart — gap in between should show ellipsis
    sentences = (
        ["Keyword Alpha here."]
        + [f"Filler {i}." for i in range(10)]
        + ["Keyword Beta here."]
    )
    text = " ".join(sentences)
    result = extract_keyword_context(text, ["Keyword Alpha", "Keyword Beta"], window=2)
    assert "..." in result


def test_extract_keyword_context_merges_overlapping_windows() -> None:
    # Two anchors close together — windows should merge, no ellipsis
    text = "Keyword one here. Middle sentence. Keyword two here. Final sentence."
    result = extract_keyword_context(text, ["Keyword one", "Keyword two"], window=5)
    assert "..." not in result
    assert "Middle sentence" in result
