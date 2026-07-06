from news_ingest.ml.schema import ML_SCHEMA_SQL


def test_ml_schema_enables_vector_and_trigram_extensions() -> None:
    assert "CREATE EXTENSION IF NOT EXISTS vector" in ML_SCHEMA_SQL
    assert "CREATE EXTENSION IF NOT EXISTS pg_trgm" in ML_SCHEMA_SQL


def test_ml_schema_has_hnsw_cosine_index() -> None:
    assert "USING hnsw" in ML_SCHEMA_SQL
    assert "vector_cosine_ops" in ML_SCHEMA_SQL
    assert "m = 16" in ML_SCHEMA_SQL
    assert "ef_construction = 64" in ML_SCHEMA_SQL
