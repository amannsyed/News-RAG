# News RAG Ingestion

Python ingestion scripts for collecting news articles from multiple APIs into PostgreSQL, then generating EmbeddingGemma vectors and GLiNER named entities for stored articles.

Supported news providers:

- NewsAPI
- GNews
- Currents
- New York Times Article Search
- NewsData.io
- TheNewsAPI

## 1. First-Time Setup

Install Python dependencies and create `.env`:

```bash
cp .env.example .env
poetry install
```

Edit `.env` and set the keys you plan to use:

```bash
NEWSAPI_API_KEY=...
GNEWS_API_KEY=...
CURRENTS_API_KEY=...
NYTIMES_API_KEY=...
NEWSDATA_API_KEY=...
THENEWSAPI_TOKEN=...
```

For embeddings and NER, also set Hugging Face/model settings:

```bash
HF_TOKEN=your-huggingface-token
MODEL_DEVICE=cpu
EMBEDDING_DTYPE=float32
EMBEDDING_MODEL_NAME=google/embeddinggemma-300m
NER_MODEL_NAME=urchade/gliner_multi-v2.1
EMBEDDING_SERVICE_URL=http://localhost:8001
NER_SERVICE_URL=http://localhost:8002
```

Before using `google/embeddinggemma-300m`, accept the model license on Hugging Face for the token in `HF_TOKEN`.

Useful shared ingestion settings:

```bash
NEWSAPI_QUERIES=technology;business;politics;science;health;AI;cybersecurity;finance;climate;startups
NEWSAPI_MAX_PAGES_PER_WINDOW=1
NEWSAPI_LANGUAGE=en
NEWSAPI_TIMEZONE=Europe/London
CURRENTS_PAGE_SIZE=20
GNEWS_PAGE_SIZE=10
NEWSDATA_PAGE_SIZE=10
NYTIMES_PAGE_SIZE=10
THENEWSAPI_PAGE_SIZE=10
```

## 2. Docker Commands

Start only PostgreSQL:

```bash
docker compose up -d postgres
```

Start PostgreSQL plus embedding and NER model services in CPU-safe mode:

```bash
docker compose up -d --build postgres embedding-worker ner-worker
```

Start with NVIDIA GPU acceleration, only after Docker GPU support works on your machine:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build postgres embedding-worker ner-worker
```

Watch model startup/download logs:

```bash
docker compose logs -f embedding-worker
```

```bash
docker compose logs -f ner-worker
```

Check service health:

```bash
curl http://localhost:8001/health
```

```bash
curl http://localhost:8002/health
```

Validate Compose config:

```bash
docker compose config
```

Stop model services when you are done:

```bash
docker compose stop embedding-worker ner-worker
```

Stop all services:

```bash
docker compose stop
```

Start all existing services again in CPU-safe mode:

```bash
docker compose up -d
```

Start all existing services again with GPU override:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

## 3. News Ingestion Commands

Run each provider separately so quota and errors are easy to control. Start with small `--max-calls` values, then increase after a successful run.

NewsAPI, last 7 days:

```bash
poetry run news-ingest --last-days 7 --max-calls 80 --sleep-seconds 2 --queries "technology;business;politics;science;health;AI;cybersecurity;finance;climate;startups"
```

NewsAPI, larger query set helper:

```bash
poetry run news-ingest-100 --last-days 7 --max-calls 100 --sleep-seconds 2 --queries "technology;business;politics;science;health;AI;cybersecurity;finance;climate;startups"
```

GNews:

```bash
poetry run gnews-ingest --last-days 7 --max-calls 50 --sleep-seconds 2 --queries "technology;business;science;health;finance;climate;sports;world;AI;cybersecurity"
```

Currents:

```bash
poetry run currents-ingest --last-days 7 --max-calls 10 --sleep-seconds 2 --queries "technology;business;science"
```

NYTimes:

```bash
poetry run nytimes-ingest --last-days 7 --max-calls 10 --sleep-seconds 2 --queries "climate;business;science"
```

NYTimes with Article Search `fq` filter:

```bash
poetry run nytimes-ingest --last-days 30 --query "climate change" --filter-query 'typeOfMaterials:News AND section.name:Climate'
```

NewsData.io latest endpoint:

```bash
poetry run newsdata-ingest --latest-only --max-calls 10 --sleep-seconds 2 --queries "technology;business;science"
```

NewsData.io archive/date windows, only if your plan includes archive access:

```bash
poetry run newsdata-ingest --last-days 1 --max-calls 10 --sleep-seconds 2 --queries "technology;business;science"
```

TheNewsAPI:

```bash
poetry run thenewsapi-ingest --last-days 7 --max-calls 20 --sleep-seconds 2 --queries "technology;business;science"
```

Common provider options:

```bash
--query "single query"
--queries "technology;business;science"
--last-days 7
--from-date 2026-06-28 --to-date 2026-07-04
--max-calls 10
--sleep-seconds 2
--refresh-existing
```

Provider help commands:

```bash
poetry run news-ingest --help
poetry run news-ingest-100 --help
poetry run gnews-ingest --help
poetry run currents-ingest --help
poetry run nytimes-ingest --help
poetry run newsdata-ingest --help
poetry run thenewsapi-ingest --help
```

## 4. Embeddings and NER Commands

The ML services process articles that are already stored in PostgreSQL.

The `embedding-worker` Docker service:

- Loads `google/embeddinggemma-300m`.
- Exposes `GET /health`, `POST /chunk`, and `POST /embed` on port `8001`.
- `/chunk` uses the EmbeddingGemma tokenizer for 500-token chunks with 50-token overlap.

The `ner-worker` Docker service:

- Loads `urchade/gliner_multi-v2.1`.
- Exposes `GET /health` and `POST /ner` on port `8002`.
- NER processing uses the embedding service `/chunk` endpoint, so embedding and NER share the same EmbeddingGemma token boundaries.

Start all required Docker services in CPU-safe mode:

```bash
docker compose up -d --build postgres embedding-worker ner-worker
```

Start all required Docker services with NVIDIA GPU acceleration:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build postgres embedding-worker ner-worker
```

Process embeddings from the host Poetry environment:

```bash
poetry run process-embeddings --limit 500 --batch-size 32 --endpoint-url http://localhost:8001
```

Process embeddings with explicit token settings:

```bash
poetry run process-embeddings \
  --limit 500 \
  --batch-size 32 \
  --max-tokens 500 \
  --overlap-tokens 50 \
  --endpoint-url http://localhost:8001
```

Process NER from the host Poetry environment:

```bash
poetry run process-ner \
  --limit 500 \
  --batch-size 16 \
  --endpoint-url http://localhost:8002 \
  --chunk-endpoint-url http://localhost:8001
```

Process NER with explicit token and threshold settings:

```bash
poetry run process-ner \
  --limit 500 \
  --batch-size 16 \
  --max-tokens 500 \
  --overlap-tokens 50 \
  --threshold 0.5 \
  --endpoint-url http://localhost:8002 \
  --chunk-endpoint-url http://localhost:8001
```

Run embedding DB worker inside Docker:

```bash
docker compose exec embedding-worker python -m news_ingest.ml.embedding_worker --limit 500 --batch-size 32
```

Run NER DB worker inside Docker:

```bash
docker compose exec ner-worker python -m news_ingest.ml.ner_worker --limit 500 --batch-size 16
```

ML help commands:

```bash
poetry run process-embeddings --help
poetry run process-ner --help
```

## 5. RAG API Commands

The RAG API exposes hybrid retrieval on port `8003`. It uses three retrieval streams and weighted Reciprocal Rank Fusion:

- Dense vector search over `article_embedding_dedup.embedding`.
- Sparse full-text search over `article_ml_documents.document_text`.
- NER/entity search over `article_entities` and `entity_aliases`.

Start the RAG API with Postgres and the embedding service:

```bash
docker compose up -d --build postgres embedding-worker rag-api
```

Start the full stack, including NER service:

```bash
docker compose up -d --build postgres embedding-worker ner-worker rag-api
```

Check RAG API health:

```bash
curl http://localhost:8003/health
```

Auth is optional for local development. If `RAG_API_TOKEN` is set, include this header on `/search`, `/conversation`, `/chat`, and `/chat/stream` calls:

```bash
-H "Authorization: Bearer $RAG_API_TOKEN"
```

Run a default hybrid search:

```bash
curl -sS -X POST http://localhost:8003/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"AI regulation in Europe","limit":5}'
```

Run the same query again to verify exact cache hits. The response includes `"cache":{"hit":true,"type":"exact"}` when served from cache:

```bash
curl -sS -X POST http://localhost:8003/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"AI regulation in Europe","limit":5}'
```

Run a semantic-heavy search override:

```bash
curl -sS -X POST http://localhost:8003/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"how AI changes bank risk management","limit":5,"weights":{"vector":0.8,"full_text":0.1,"entity":0.1}}'
```

Run an entity-heavy search override:

```bash
curl -sS -X POST http://localhost:8003/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Anthropic Europe AI regulation","limit":5,"weights":{"vector":0.2,"full_text":0.4,"entity":0.4}}'
```

Stream a citation-grounded extractive conversation response with Server-Sent Events:

```bash
curl -N -X POST http://localhost:8003/conversation \
  -H 'Content-Type: application/json' \
  -d '{"conversation_id":"demo-ai-regulation","message":"What is happening with AI regulation in Europe?","limit":5}'
```

Run a Claude-backed chat answer. By default this uses Bedrock when `AWS_BEDROCK_KEY` exists, otherwise Vertex AI. Set `CHAT_PROVIDER=bedrock` or `CHAT_PROVIDER=vertex` to force one provider:

```bash
curl -sS -X POST http://localhost:8003/chat \
  -H 'Content-Type: application/json' \
  -d '{"conversation_id":"claude-demo","message":"What is happening with AI regulation in Europe?","limit":5}'
```

Run chat with indexed web-search citations added to the RAG citations:

```bash
curl -sS -X POST http://localhost:8003/chat \
  -H 'Content-Type: application/json' \
  -d '{"conversation_id":"web-search-demo","message":"What is the latest on AI regulation in Europe?","limit":3,"web_search":true,"web_search_limit":3}'
```

Stream a Claude-backed chat answer with Server-Sent Events:

```bash
curl -N -X POST http://localhost:8003/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"conversation_id":"claude-demo-stream","message":"What is happening with AI regulation in Europe?","limit":5}'
```

The `/conversation` stream emits extractive `retrieval`, `answer_chunk`, `complete`, and `error` events. `/chat` and `/chat/stream` first retrieve citations, then call Claude through Bedrock or Vertex AI and store user/assistant turns in `rag_conversations`.

AWS Bedrock environment settings:

```bash
CHAT_PROVIDER=bedrock
AWS_BEDROCK_REGION=us-east-1
AWS_BEDROCK_MODEL_ID=global.anthropic.claude-haiku-4-5-20251001-v1:0
AWS_BEDROCK_KEY={"aws_access_key_id":"...","aws_secret_access_key":"...","region":"us-east-1"}
```

You can also use standard AWS variables instead of `AWS_BEDROCK_KEY`: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optional `AWS_SESSION_TOKEN`. `AWS_BEDROCK_KEY` may also be `access_key_id:secret_access_key[:session_token]`.

Claude on Vertex AI environment settings:

```bash
VERTEX_PROJECT_ID=your-gcp-project-id
VERTEX_LOCATION=global
CLAUDE_MODEL=claude-haiku-4-5@20251001
CLAUDE_MAX_TOKENS=1200
CLAUDE_TEMPERATURE=0.2
```

For credentials, use Application Default Credentials, set `GOOGLE_APPLICATION_CREDENTIALS` to a service-account JSON path, or put service-account JSON directly in `GOOGLE_APPLICATION_CREDENTIALS_JSON` / `VERTEX_CREDENTIALS_JSON` for Docker runs.

Run the API locally with Poetry instead of Docker:

```bash
EMBEDDING_SERVICE_URL=http://localhost:8001 poetry run rag-api
```

RRF defaults:

- `rrf_k`: `60`
- default env weights: `RAG_VECTOR_WEIGHT=0.6`, `RAG_FULL_TEXT_WEIGHT=0.2`, `RAG_ENTITY_WEIGHT=0.2`
- auth: `RAG_API_TOKEN`; empty disables auth
- rate limit: `RAG_RATE_LIMIT_PER_MINUTE=60`, `RAG_RATE_LIMIT_BURST=10`
- cache: `RAG_CACHE_TTL_SECONDS=3600`, `RAG_SEMANTIC_CACHE_THRESHOLD=0.95`
- chat provider: `CHAT_PROVIDER=bedrock` or `CHAT_PROVIDER=vertex`
- Bedrock: `AWS_BEDROCK_REGION`, `AWS_BEDROCK_MODEL_ID`, `AWS_BEDROCK_KEY` or standard AWS credential env vars
- web search: request fields `web_search=true`, `web_search_limit`; env `WEB_SEARCH_PROVIDER=bing_news`, `WEB_SEARCH_TTL_SECONDS=21600`
- Claude on Vertex: `VERTEX_PROJECT_ID`, `VERTEX_LOCATION`, `CLAUDE_MODEL`, `GOOGLE_APPLICATION_CREDENTIALS` or `GOOGLE_APPLICATION_CREDENTIALS_JSON`
- no entity candidates: dynamic weights bias vector and full-text
- entity candidates found: dynamic weights bias vector and entity search

The `/search` response includes article IDs, titles, source URLs, fused RRF scores, per-stream ranks/scores, matched snippets, chunk/document IDs, character spans, entity match details, and cache hit metadata.

## 6. Database Check Commands

Provider counts:

```bash
docker compose exec -T postgres psql -U news_rag -d news_rag -c "SELECT provider, COUNT(*) FROM article_metadata GROUP BY provider ORDER BY provider;"
```

Date coverage:

```bash
docker compose exec -T postgres psql -U news_rag -d news_rag -c "SELECT provider, window_label, COUNT(*) FROM article_metadata GROUP BY provider, window_label ORDER BY provider, window_label;"
```

Source coverage:

```bash
docker compose exec -T postgres psql -U news_rag -d news_rag -c "SELECT provider, COUNT(DISTINCT source_name) AS sources FROM article_metadata GROUP BY provider ORDER BY provider;"
```

Embedding row count:

```bash
docker compose exec -T postgres psql -U news_rag -d news_rag -c "SELECT COUNT(*) FROM article_embedding;"
```

Embedding dedup row count:

```bash
docker compose exec -T postgres psql -U news_rag -d news_rag -c "SELECT COUNT(*) FROM article_embedding_dedup;"
```

Entity counts by type:

```bash
docker compose exec -T postgres psql -U news_rag -d news_rag -c "SELECT et.label, COUNT(*) FROM article_entities ae JOIN entity_types et ON et.id = ae.entity_type_id GROUP BY et.label ORDER BY et.label;"
```

Top entities by salience:

```bash
docker compose exec -T postgres psql -U news_rag -d news_rag -c "SELECT et.label, ae.normalized_text, COUNT(*) AS mentions, MAX(ae.salience) AS max_salience FROM article_entities ae JOIN entity_types et ON et.id = ae.entity_type_id GROUP BY et.label, ae.normalized_text ORDER BY max_salience DESC LIMIT 25;"
```

Postgres shell:

```bash
docker compose exec postgres psql -U news_rag -d news_rag
```

## 7. Logs and Troubleshooting Commands

Application logs:

```bash
tail -n 120 logs/info.log
```

```bash
tail -n 120 logs/warning.log
```

```bash
tail -n 120 logs/error.log
```

Docker logs:

```bash
docker compose logs -f postgres
```

```bash
docker compose logs -f embedding-worker
```

```bash
docker compose logs -f ner-worker
```

```bash
docker compose logs -f rag-api
```

Container status:

```bash
docker compose ps
```

GPU visibility from Docker, after starting with `docker-compose.gpu.yml`:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec embedding-worker python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Health endpoint checks:

```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
```

## 8. Storage Model

News ingestion tables:

- `article_metadata`: URL, source, title, description, provider, query/window metadata, published/fetched timestamps.
- `article_contents`: content text, content hash, and raw provider JSON payload.

ML tables:

- `article_ml_documents`: stable text snapshots used for chunking.
- `article_embedding_dedup`: unique chunk vectors keyed by content hash and model.
- `article_embedding`: article/chunk/span mappings to deduped vectors.
- `entity_types`: entity label definitions.
- `entity_aliases`: normalized aliases and canonical forms.
- `ner_content_dedup`: cached GLiNER payloads for duplicate chunks.
- `article_entities`: entity mentions, spans, confidence, and salience.

Deduplication uses:

- normalized URL hash
- non-empty normalized article content hash
- provider/query/window skip checks before API calls
- 30-day chunk content-hash reuse before embedding model calls
- NER chunk content-hash cache before GLiNER calls

## 9. Test and Validation Commands

Run tests:

```bash
poetry run pytest -q
```

Compile source and tests:

```bash
poetry run python -m compileall -q src tests
```

Validate Docker Compose:

```bash
docker compose config
```

Install/update local Poetry entrypoints after changing scripts:

```bash
poetry install
```

## Evaluation And Monitoring

This is intentionally lightweight and local. The RAG API writes one row per `/search`, `/conversation`, `/chat`, and `/chat/stream` request into `rag_request_metrics`. It records endpoint, user/conversation, latency, status, retrieval count, citation count, web citation count, model/provider, cache type, and any error text.

Start or restart the API so the metrics table is created:

```bash
docker compose up -d --build rag-api
```

View monitoring summary for the last 24 hours:

```bash
curl -sS "http://localhost:8003/monitoring/summary?window_hours=24" \
  -H "Authorization: Bearer $RAG_API_TOKEN"
```

Run a quick deterministic evaluation over recent chat answers:

```bash
poetry run rag-eval --limit 25 --pretty
```

Or call it through the API:

```bash
curl -sS "http://localhost:8003/evaluation/recent?limit=25&user_id=user_id" \
  -H "Authorization: Bearer $RAG_API_TOKEN"
```

The evaluator is not an LLM judge. It is a fast smoke-test layer that checks citation coverage, answer length, retrieval payload presence, web citation count, cache type, and warning counts such as missing inline citation markers. Use it for regression checks and operational visibility, not final answer grading.

Useful DB checks:

```bash
docker compose exec -T postgres psql -U news_rag -d news_rag -c "SELECT endpoint, status, COUNT(*), ROUND(AVG(latency_ms)::numeric, 1) AS avg_ms FROM rag_request_metrics GROUP BY endpoint, status ORDER BY endpoint, status;"

docker compose exec -T postgres psql -U news_rag -d news_rag -c "SELECT created_at, endpoint, status, latency_ms, citation_count, error FROM rag_request_metrics ORDER BY created_at DESC LIMIT 20;"
```

## Standalone MCP Server

The project includes a FastMCP wrapper around the RAG API. It exposes the main API features as MCP tools:

- `rag_health`
- `search_news`
- `chat_news`
- `chat_news_stream`
- `extractive_conversation_stream`
- `list_conversations`
- `get_conversation`
- `delete_conversation`
- `monitoring_summary`
- `evaluate_recent_chats`

Start the required backend first:

```bash
docker compose up -d postgres embedding-worker rag-api
```

Run the MCP server locally over stdio, which is the normal mode for desktop MCP clients:

```bash
poetry install
MCP_RAG_API_URL=http://localhost:8003 MCP_RAG_API_TOKEN="$RAG_API_TOKEN" poetry run news-rag-mcp
```

Example MCP client config for a stdio client:

```json
{
  "mcpServers": {
    "news-rag": {
      "command": "poetry",
      "args": ["run", "news-rag-mcp"],
      "cwd": "/projects/GitHub/News RAG",
      "env": {
        "MCP_RAG_API_URL": "http://localhost:8003",
        "MCP_RAG_API_TOKEN": "your-rag-api-token"
      }
    }
  }
}
```

Run the MCP server as a standalone HTTP service in Docker:

```bash
docker compose up -d --build news-rag-mcp
```

The HTTP MCP endpoint is exposed at:

```text
http://localhost:8004/mcp
```

You can also run HTTP mode locally without Docker:

```bash
poetry run news-rag-mcp --transport http --host 0.0.0.0 --port 8004
```

MCP environment variables:

```bash
MCP_RAG_API_URL=http://localhost:8003
MCP_RAG_API_TOKEN=your-rag-api-token
MCP_TRANSPORT=stdio
MCP_HOST=0.0.0.0
MCP_PORT=8004
MCP_PATH=/mcp
```

## Web UI

Start the backend services:

```bash
docker compose up -d postgres embedding-worker rag-api
```

Run the React UI:

```bash
cd ui
npm install
npm run dev
```

Open the Vite URL, normally `http://localhost:5173`. The UI proxies `/api/*` to the RAG API on `http://localhost:8003`. Conversation history is currently scoped to the default `user_id` value `user_id`. Chat retrieval defaults to 20 indexed RAG articles, and web search requests default to 20 live web results when enabled.

The UI no longer hard-codes a bearer token. Paste `RAG_API_TOKEN` into the sidebar token field, or for local-only development create `ui/.env.local` with:

```bash
VITE_RAG_API_TOKEN=your-local-rag-token
```

Do not use `VITE_RAG_API_TOKEN` for production secrets because Vite exposes it to the browser bundle.


Optional Claude thinking budgets for the main answer call only:

```bash
CLAUDE_THINKING_LOW_TOKENS=512
CLAUDE_THINKING_MEDIUM_TOKENS=1024
CLAUDE_THINKING_MAX_TOKENS=2048
```

Thinking is not used for background summarization or follow-up generation calls.

Chat memory behavior:

- `/chat` sends the rolling conversation summary plus the last 10 stored messages, which is up to 5 user/assistant turns, into the answer prompt.
- After each answer, it makes a separate summarization call and stores the updated summary in `rag_conversation_summaries`.
- It also makes a separate follow-up generation call and returns `follow_up_questions` in the response.
- The UI renders follow-up questions as chips; clicking one fills the chat input but does not automatically send it.

