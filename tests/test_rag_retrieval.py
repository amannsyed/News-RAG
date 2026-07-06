from news_ingest.ml.rag_cache import cache_fingerprint, to_jsonable
from news_ingest.ml.rag_retrieval import RetrievalWeights, StreamHit, dynamic_weights, rrf_fuse


def test_dynamic_weights_bias_semantic_without_entities() -> None:
    weights = dynamic_weights(base=RetrievalWeights(), has_entities=False)
    assert weights.vector > weights.full_text > weights.entity
    assert round(weights.vector + weights.full_text + weights.entity, 6) == 1.0


def test_dynamic_weights_bias_entity_when_entities_present() -> None:
    weights = dynamic_weights(base=RetrievalWeights(), has_entities=True)
    assert weights.vector > weights.entity > weights.full_text
    assert round(weights.vector + weights.full_text + weights.entity, 6) == 1.0


def test_dynamic_weights_request_override_wins() -> None:
    weights = dynamic_weights(base=RetrievalWeights(), has_entities=True, request_override=RetrievalWeights(vector=0.2, full_text=0.4, entity=0.4))
    assert weights == RetrievalWeights(vector=0.2, full_text=0.4, entity=0.4)


def test_rrf_fuse_combines_stream_ranks_by_article() -> None:
    streams = {
        "vector": [StreamHit(article_id=1, rank=1, score=0.9, stream="vector"), StreamHit(article_id=2, rank=2, score=0.8, stream="vector")],
        "full_text": [StreamHit(article_id=2, rank=1, score=1.2, stream="full_text")],
        "entity": [StreamHit(article_id=3, rank=1, score=2.0, stream="entity")],
    }

    results = rrf_fuse(streams, weights=RetrievalWeights(vector=0.6, full_text=0.3, entity=0.1), k=60, limit=3)

    assert results[0].article_id == 2
    assert results[0].stream_ranks == {"vector": 2, "full_text": 1}
    assert {result.article_id for result in results} == {1, 2, 3}


def test_cache_fingerprint_is_stable_for_weight_order() -> None:
    left = cache_fingerprint(limit=5, rrf_k=60, weights={"vector": 0.6, "full_text": 0.2, "entity": 0.2})
    right = cache_fingerprint(limit=5, rrf_k=60, weights={"entity": 0.2, "vector": 0.6, "full_text": 0.2})

    assert left == right


def test_to_jsonable_converts_nested_dataclasses() -> None:
    hit = StreamHit(article_id=1, rank=1, score=0.9, stream="vector")

    assert to_jsonable({"hits": [hit]})["hits"][0]["article_id"] == 1
