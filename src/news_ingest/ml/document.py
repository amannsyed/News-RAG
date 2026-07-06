"""Shared document persistence helpers used by both embedding and NER workers."""
from __future__ import annotations

from typing import Any

from news_ingest.ml.text import build_article_document, stable_hash


DOCUMENT_SCOPE = "ml-v1"


def ensure_document(conn, *, article: dict[str, Any], document_text: str) -> int:
    """Upsert an article ML document row and return its id.

    Both the embedding worker and the NER worker need to persist the document
    text before writing their respective derived rows.  Keeping the SQL in one
    place ensures the two workers stay in sync if the table schema ever changes.
    """
    document_hash = stable_hash(document_text)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO article_ml_documents(article_id, document_text, document_hash, model_scope)
            VALUES (%(article_id)s, %(document_text)s, %(document_hash)s, %(model_scope)s)
            ON CONFLICT (article_id, model_scope) DO UPDATE SET
                document_text = EXCLUDED.document_text,
                document_hash = EXCLUDED.document_hash,
                updated_at = now()
            RETURNING id;
            """,
            {
                "article_id": article["id"],
                "document_text": document_text,
                "document_hash": document_hash,
                "model_scope": DOCUMENT_SCOPE,
            },
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("document upsert did not return a row")
    return int(row["id"])


def get_article_document_text(article: dict[str, Any]) -> str:
    """Build the canonical document text from an article dict."""
    return build_article_document(
        title=article.get("title"),
        description=article.get("description"),
        content=article.get("content"),
    )
