from __future__ import annotations

import psycopg


ML_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS article_ml_documents (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT NOT NULL REFERENCES article_metadata(id) ON DELETE CASCADE,
    document_text TEXT NOT NULL,
    document_hash CHAR(64) NOT NULL,
    model_scope TEXT NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (article_id, model_scope)
);

CREATE TABLE IF NOT EXISTS article_embedding_dedup (
    id BIGSERIAL PRIMARY KEY,
    content_hash CHAR(64) NOT NULL,
    embedding vector(768) NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 1,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (content_hash, model_name, model_version)
);

CREATE TABLE IF NOT EXISTS article_embedding (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT NOT NULL REFERENCES article_metadata(id) ON DELETE CASCADE,
    document_id BIGINT NOT NULL REFERENCES article_ml_documents(id) ON DELETE CASCADE,
    embedding_dedup_id BIGINT NOT NULL REFERENCES article_embedding_dedup(id) ON DELETE RESTRICT,
    chunk_index INTEGER NOT NULL,
    content_hash CHAR(64) NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    load_date DATE NOT NULL DEFAULT CURRENT_DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (article_id, content_hash, model_name, model_version)
);

CREATE INDEX IF NOT EXISTS idx_article_embedding_article_id ON article_embedding(article_id);
CREATE INDEX IF NOT EXISTS idx_article_embedding_load_date ON article_embedding(load_date);
CREATE INDEX IF NOT EXISTS idx_article_embedding_dedup_seen ON article_embedding_dedup(content_hash, last_seen_at DESC);
CREATE INDEX IF NOT EXISTS idx_article_embedding_dedup_hnsw
    ON article_embedding_dedup USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS entity_types (
    id SMALLSERIAL PRIMARY KEY,
    label TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO entity_types(label, description) VALUES
    ('PERSON', 'People and named individuals'),
    ('COMPANY', 'Companies and commercial brands'),
    ('ORGANIZATION', 'Organizations, agencies, institutions, and groups'),
    ('REGULATION', 'Laws, regulations, policies, standards, and legal frameworks'),
    ('LOCATION', 'Geographic places and geopolitical locations'),
    ('FINANCIAL_METRIC', 'Financial values, metrics, market indicators, and economic measures')
ON CONFLICT (label) DO NOTHING;

CREATE TABLE IF NOT EXISTS entity_aliases (
    id BIGSERIAL PRIMARY KEY,
    entity_type_id SMALLINT NOT NULL REFERENCES entity_types(id) ON DELETE RESTRICT,
    alias TEXT NOT NULL,
    normalized_alias TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (entity_type_id, normalized_alias)
);

CREATE INDEX IF NOT EXISTS idx_entity_aliases_trgm
    ON entity_aliases USING gin (normalized_alias gin_trgm_ops);

CREATE TABLE IF NOT EXISTS ner_content_dedup (
    id BIGSERIAL PRIMARY KEY,
    content_hash CHAR(64) NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    labels TEXT[] NOT NULL,
    payload JSONB NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 1,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (content_hash, model_name, model_version, labels)
);

CREATE TABLE IF NOT EXISTS article_entities (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT NOT NULL REFERENCES article_metadata(id) ON DELETE CASCADE,
    document_id BIGINT NOT NULL REFERENCES article_ml_documents(id) ON DELETE CASCADE,
    entity_type_id SMALLINT NOT NULL REFERENCES entity_types(id) ON DELETE RESTRICT,
    alias_id BIGINT REFERENCES entity_aliases(id) ON DELETE SET NULL,
    ner_dedup_id BIGINT REFERENCES ner_content_dedup(id) ON DELETE SET NULL,
    entity_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    salience DOUBLE PRECISION NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    load_date DATE NOT NULL DEFAULT CURRENT_DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (article_id, entity_type_id, normalized_text, char_start, char_end, model_name, model_version)
);

CREATE INDEX IF NOT EXISTS idx_article_entities_article_id ON article_entities(article_id);
CREATE INDEX IF NOT EXISTS idx_article_entities_type_salience ON article_entities(entity_type_id, salience DESC);
CREATE INDEX IF NOT EXISTS idx_article_entities_load_date ON article_entities(load_date);
CREATE INDEX IF NOT EXISTS idx_article_ml_documents_fts
    ON article_ml_documents USING gin (to_tsvector('english', document_text));

CREATE INDEX IF NOT EXISTS idx_article_entities_normalized_text_trgm
    ON article_entities USING gin (normalized_text gin_trgm_ops);



CREATE TABLE IF NOT EXISTS web_search_documents (
    id BIGSERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    query_hash CHAR(64) NOT NULL,
    provider TEXT NOT NULL DEFAULT 'bing_news',
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    url_hash CHAR(64) NOT NULL,
    snippet TEXT,
    document_text TEXT NOT NULL,
    published_at TIMESTAMPTZ,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    score DOUBLE PRECISION,
    UNIQUE (url_hash, query_hash)
);

CREATE INDEX IF NOT EXISTS idx_web_search_documents_query_hash
    ON web_search_documents(query_hash, fetched_at DESC);
CREATE INDEX IF NOT EXISTS idx_web_search_documents_url_hash
    ON web_search_documents(url_hash);
CREATE INDEX IF NOT EXISTS idx_web_search_documents_fts
    ON web_search_documents USING gin (to_tsvector('english', document_text));

CREATE TABLE IF NOT EXISTS rag_search_cache (
    id BIGSERIAL PRIMARY KEY,
    query_hash CHAR(64) NOT NULL,
    query TEXT NOT NULL,
    query_embedding vector(768) NOT NULL,
    request_fingerprint CHAR(64) NOT NULL,
    response JSONB NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ NOT NULL,
    last_hit_at TIMESTAMPTZ,
    UNIQUE (query_hash, request_fingerprint)
);

CREATE INDEX IF NOT EXISTS idx_rag_search_cache_query_hash
    ON rag_search_cache(query_hash, request_fingerprint, expires_at);
CREATE INDEX IF NOT EXISTS idx_rag_search_cache_expires_at
    ON rag_search_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_rag_search_cache_embedding_hnsw
    ON rag_search_cache USING hnsw (query_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS rag_conversations (
    id BIGSERIAL PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_conversations_conversation_id_created
    ON rag_conversations(conversation_id, created_at DESC);

CREATE TABLE IF NOT EXISTS rag_conversation_summaries (
    conversation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'user_id',
    summary TEXT NOT NULL,
    turn_count INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_conversation_summaries_user_updated
    ON rag_conversation_summaries(user_id, updated_at DESC);


CREATE TABLE IF NOT EXISTS rag_request_metrics (
    id BIGSERIAL PRIMARY KEY,
    endpoint TEXT NOT NULL,
    conversation_id TEXT,
    user_id TEXT NOT NULL DEFAULT 'user_id',
    query TEXT,
    provider TEXT,
    model TEXT,
    status TEXT NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    retrieval_count INTEGER NOT NULL DEFAULT 0,
    citation_count INTEGER NOT NULL DEFAULT 0,
    web_citation_count INTEGER NOT NULL DEFAULT 0,
    answer_chars INTEGER NOT NULL DEFAULT 0,
    cache_type TEXT,
    error TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_request_metrics_created
    ON rag_request_metrics(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rag_request_metrics_endpoint_created
    ON rag_request_metrics(endpoint, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_rag_request_metrics_user_created
    ON rag_request_metrics(user_id, created_at DESC);

"""


def ensure_ml_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(ML_SCHEMA_SQL)
    conn.commit()
