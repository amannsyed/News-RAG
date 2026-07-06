# News Ingest Package

This package contains the ingestion layer and shared infrastructure for the News RAG system.

## Responsibilities

- Read configuration from environment variables and `.env`.
- Fetch articles from multiple news providers.
- Normalize provider payloads into a shared database shape.
- Deduplicate articles by URL/content hash.
- Store metadata, article content, raw payloads, logs, and enrichment-ready records.

## Provider Modules

- `ingest.py`, `newsapi_client.py`: NewsAPI ingestion.
- `currents_ingest.py`, `currents_client.py`: Currents API ingestion.
- `gnews_ingest.py`, `gnews_client.py`: GNews ingestion.
- `nytimes_ingest.py`, `nytimes_client.py`: NYTimes Article Search ingestion.
- `newsdata_ingest.py`, `newsdata_client.py`: NewsData ingestion.
- `thenewsapi_ingest.py`, `thenewsapi_client.py`: TheNewsAPI ingestion.

## Shared Modules

- `config.py`: environment and settings loader.
- `db.py`: PostgreSQL schema setup and persistence helpers.
- `dates.py`: date-window helpers for ingestion.
- `logging_config.py`: file and console logging setup.
- `hundred_ingest.py`: repeated/paced ingestion helper.

## Example Commands

```bash
poetry run news-ingest --last-days 1 --max-calls 10
poetry run gnews-ingest --last-days 7 --queries "technology;business;science"
poetry run nytimes-ingest --last-days 7 --queries "climate;business;science"
poetry run newsdata-ingest --latest-only --queries "technology;business"
poetry run thenewsapi-ingest --last-days 1 --queries "technology;business"
```
