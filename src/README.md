# Source

Python source code for the News RAG backend, ingestion jobs, ML workers, RAG API, and MCP integration.

The package uses the `src/` layout and is installed by Poetry as `news_ingest`.

## Main Package

- `news_ingest/`: ingestion, database, logging, ML enrichment, retrieval, chat, monitoring, and MCP server code.

## Useful Commands

Run all tests:

```bash
poetry run pytest -q
```

Compile-check the Python source:

```bash
poetry run python -m compileall -q src tests
```

Run the RAG API locally:

```bash
poetry run rag-api
```
