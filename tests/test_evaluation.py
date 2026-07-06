from datetime import datetime, timezone

from news_ingest.ml.evaluation import _evaluate_row, _summary


def test_evaluate_row_scores_citation_coverage() -> None:
    row = {
        "conversation_id": "c1",
        "created_at": datetime(2026, 7, 5, tzinfo=timezone.utc),
        "question": "What happened?",
        "answer": "The story cites two sources [1] and [2]. " * 8,
        "payload": {
            "citations": [
                {"citation_index": 1, "source_type": "rag_article"},
                {"citation_index": 2, "source_type": "web_search"},
            ],
            "retrieval": {"results": [{"article_id": 1}], "cache": {"type": "exact"}},
        },
    }

    result = _evaluate_row(row)

    assert result.citation_count == 2
    assert result.referenced_citation_count == 2
    assert result.citation_coverage == 1.0
    assert result.web_citation_count == 1
    assert result.retrieval_count == 1
    assert result.cache_type == "exact"
    assert result.warnings == []


def test_evaluate_row_warns_when_sources_not_cited() -> None:
    row = {
        "conversation_id": "c1",
        "created_at": datetime(2026, 7, 5, tzinfo=timezone.utc),
        "question": "What happened?",
        "answer": "No inline markers here.",
        "payload": {"citations": [{"citation_index": 1}], "retrieval": {"results": []}},
    }

    result = _evaluate_row(row)

    assert "answer_has_sources_but_no_inline_citation_markers" in result.warnings
    assert "answer_is_short" in result.warnings
    assert "no_retrieval_results_in_payload" in result.warnings


def test_summary_aggregates_warning_counts() -> None:
    row = {
        "conversation_id": "c1",
        "created_at": datetime(2026, 7, 5, tzinfo=timezone.utc),
        "question": "Q",
        "answer": "Short",
        "payload": {"citations": [{"citation_index": 1}], "retrieval": {"results": []}},
    }

    report = _summary([_evaluate_row(row)])

    assert report["count"] == 1
    assert report["warning_counts"]["answer_is_short"] == 1
