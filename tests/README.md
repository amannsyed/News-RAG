# Tests

Pytest suite for ingestion, database helpers, ML workers, retrieval, API security, web search, MCP tooling, and monitoring/evaluation.

## Test Areas

- Provider ingestion: `test_ingest.py`, `test_currents.py`, `test_gnews.py`, `test_nytimes.py`, `test_newsdata.py`, `test_thenewsapi.py`.
- Database and date helpers: `test_db.py`, `test_dates.py`.
- ML processing: `test_embedding_worker.py`, `test_ner_processing.py`, `test_ml_schema.py`, `test_ml_text.py`.
- RAG API and retrieval: `test_rag_api.py`, `test_rag_retrieval.py`, `test_rag_security.py`, `test_web_search.py`.
- LLM/MCP/evaluation: `test_claude_vertex.py`, `test_mcp_server.py`, `test_evaluation.py`.

## Commands

```bash
poetry run pytest -q
poetry run python -m compileall -q src tests
```

The tests are designed to run without real API keys by using mocks, fake payloads, and isolated database helpers.
