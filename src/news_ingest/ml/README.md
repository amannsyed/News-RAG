# ML, Retrieval, And RAG

This package contains the enrichment workers, model services, hybrid retrieval, chat API, MCP server, and lightweight monitoring/evaluation code.

## Model Services

- `embedding_service.py`: FastAPI service that exposes `/embed` for `google/embeddinggemma-300m`.
- `ner_service.py`: FastAPI service that exposes `/extract` for GLiNER NER.
- `http_client.py`: small JSON HTTP client used by workers.

## Background Workers

- `embedding_worker.py`: chunks articles, deduplicates chunks, calls the embedding service, and writes pgvector rows.
- `ner_worker.py`: chunks articles, deduplicates NER work, calls the NER service, and writes entity rows.
- `text.py`: tokenizer-aware chunking and text-span helpers.
- `ner.py`: entity normalization, overlap handling, salience, and alias logic.
- `document.py`: document text assembly helpers.

## RAG API

- `rag_api.py`: FastAPI app for search, chat, conversation management, monitoring, and evaluation endpoints.
- `rag_retrieval.py`: vector, full-text, entity retrieval, and weighted RRF fusion.
- `rag_cache.py`: exact and semantic cache helpers.
- `rag_security.py`: optional bearer-token auth and lightweight rate limiting.
- `web_search.py`: optional web-search document indexing and citation support.
- `schema.py`: request/response models.

## LLM And Integrations

- `claude_vertex.py`: Claude client abstraction for Vertex AI and AWS Bedrock.
- `mcp_server.py`: standalone FastMCP server exposing RAG tools.
- `evaluation.py`: simple recent-response evaluation and monitoring summaries.
- `rag_cli.py`: CLI entry points for evaluation helpers.

## Example Commands

```bash
poetry run process-embeddings --limit 100 --batch-size 4 --endpoint-url http://localhost:8001
poetry run process-ner --limit 100 --batch-size 4 --endpoint-url http://localhost:8002 --chunk-endpoint-url http://localhost:8001
poetry run rag-api
poetry run news-rag-mcp
poetry run rag-eval --limit 25 --pretty
```
