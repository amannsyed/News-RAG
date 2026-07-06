from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


NER_LABELS = ("PERSON", "COMPANY", "ORGANIZATION", "REGULATION", "LOCATION", "FINANCIAL_METRIC")


@dataclass(frozen=True)
class EntityMention:
    text: str
    label: str
    start: int
    end: int
    confidence: float


def normalize_entity_text(value: str) -> str:
    stripped = value.strip().strip(".,;:'\"()[]{}")
    return re.sub(r"\s+", " ", stripped).lower()


def deduplicate_overlaps(entities: list[EntityMention]) -> list[EntityMention]:
    ordered = sorted(entities, key=lambda item: (item.start, item.end))
    kept: list[EntityMention] = []
    for entity in ordered:
        replacement_index = None
        should_keep = True
        for index, existing in enumerate(kept):
            overlaps = entity.start < existing.end and existing.start < entity.end
            if not overlaps:
                continue
            entity_width = entity.end - entity.start
            existing_width = existing.end - existing.start
            if (entity.confidence, entity_width) > (existing.confidence, existing_width):
                replacement_index = index
            else:
                should_keep = False
            break
        if replacement_index is not None:
            kept[replacement_index] = entity
        elif should_keep:
            kept.append(entity)
    return sorted(kept, key=lambda item: (item.start, item.end))


def salience_score(*, confidence: float, position: int, article_length: int, beta: float = 4.0) -> float:
    if article_length <= 0:
        return confidence
    return confidence * math.exp(-beta * (position / article_length))


def entity_from_payload(raw: dict[str, Any], *, chunk_offset: int = 0) -> EntityMention | None:
    text = str(raw.get("text") or "").strip()
    label = str(raw.get("label") or "").upper()
    start = raw.get("start")
    end = raw.get("end")
    confidence = raw.get("score", raw.get("confidence", 0))
    if not text or label not in NER_LABELS or start is None or end is None:
        return None
    return EntityMention(text=text, label=label, start=int(start) + chunk_offset, end=int(end) + chunk_offset, confidence=float(confidence))
