from news_ingest.ml.ner import EntityMention, deduplicate_overlaps, entity_from_payload, normalize_entity_text, salience_score


def test_normalize_entity_text_trims_punctuation_and_case() -> None:
    assert normalize_entity_text('  Apple Inc., ') == "apple inc"


def test_deduplicate_overlaps_keeps_higher_confidence() -> None:
    entities = [
        EntityMention("Apple", "COMPANY", 0, 5, 0.70),
        EntityMention("Apple Inc", "COMPANY", 0, 9, 0.80),
        EntityMention("London", "LOCATION", 20, 26, 0.60),
    ]

    deduped = deduplicate_overlaps(entities)

    assert [entity.text for entity in deduped] == ["Apple Inc", "London"]


def test_salience_penalizes_later_mentions() -> None:
    early = salience_score(confidence=1.0, position=0, article_length=100)
    late = salience_score(confidence=1.0, position=80, article_length=100)
    assert early > late


def test_entity_from_payload_adds_chunk_offset_and_filters_labels() -> None:
    entity = entity_from_payload({"text": "Apple", "label": "company", "score": 0.9, "start": 2, "end": 7}, chunk_offset=10)
    ignored = entity_from_payload({"text": "Monday", "label": "date", "score": 0.9, "start": 0, "end": 6})

    assert entity == EntityMention("Apple", "COMPANY", 12, 17, 0.9)
    assert ignored is None
