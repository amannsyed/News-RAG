from __future__ import annotations

import os
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    newsapi_api_key: str
    currents_api_key: str | None
    gnews_api_key: str | None
    nytimes_api_key: str | None
    newsdata_api_key: str | None
    thenewsapi_token: str | None
    database_url: str
    query: str
    queries: tuple[str, ...]
    max_calls: int = 1000
    page_size: int = 100
    max_pages_per_window: int = 1
    currents_page_size: int = 20
    gnews_page_size: int = 10
    nytimes_page_size: int = 10
    newsdata_page_size: int = 10
    thenewsapi_page_size: int = 10
    language: str | None = "en"
    sort_by: str = "publishedAt"
    timezone: str = "Europe/London"

    @property
    def zoneinfo(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


def load_settings(*, require_newsapi: bool = True, require_query: bool = True, require_currents: bool = False, require_gnews: bool = False, require_nytimes: bool = False, require_newsdata: bool = False, require_thenewsapi: bool = False) -> Settings:
    load_dotenv()

    api_key = os.getenv("NEWSAPI_API_KEY", "").strip()
    currents_api_key = os.getenv("CURRENTS_API_KEY", "").strip() or None
    gnews_api_key = os.getenv("GNEWS_API_KEY", "").strip() or None
    nytimes_api_key = os.getenv("NYTIMES_API_KEY", "").strip() or None
    newsdata_api_key = os.getenv("NEWSDATA_API_KEY", "").strip() or None
    thenewsapi_token = os.getenv("THENEWSAPI_TOKEN", "").strip() or None
    database_url = os.getenv("DATABASE_URL", "").strip()
    query = os.getenv("NEWSAPI_QUERY", "").strip()
    queries = parse_queries(os.getenv("NEWSAPI_QUERIES", ""))

    required_values = {
        "DATABASE_URL": database_url,
    }
    if require_query:
        required_values["NEWSAPI_QUERY or NEWSAPI_QUERIES"] = query or queries
    if require_newsapi:
        required_values["NEWSAPI_API_KEY"] = api_key
    if require_currents:
        required_values["CURRENTS_API_KEY"] = currents_api_key
    if require_gnews:
        required_values["GNEWS_API_KEY"] = gnews_api_key
    if require_nytimes:
        required_values["NYTIMES_API_KEY"] = nytimes_api_key
    if require_newsdata:
        required_values["NEWSDATA_API_KEY"] = newsdata_api_key
    if require_thenewsapi:
        required_values["THENEWSAPI_TOKEN"] = thenewsapi_token

    missing = [name for name, value in required_values.items() if not value]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    page_size = int(os.getenv("NEWSAPI_PAGE_SIZE", "100"))
    if page_size < 1 or page_size > 100:
        raise ValueError("NEWSAPI_PAGE_SIZE must be between 1 and 100")

    max_calls = int(os.getenv("NEWSAPI_MAX_CALLS", "1000"))
    if max_calls < 1:
        raise ValueError("NEWSAPI_MAX_CALLS must be at least 1")

    max_pages_per_window = int(os.getenv("NEWSAPI_MAX_PAGES_PER_WINDOW", "1"))
    if max_pages_per_window < 1:
        raise ValueError("NEWSAPI_MAX_PAGES_PER_WINDOW must be at least 1")

    currents_page_size = int(os.getenv("CURRENTS_PAGE_SIZE", "20"))
    if currents_page_size < 1 or currents_page_size > 20:
        raise ValueError("CURRENTS_PAGE_SIZE must be between 1 and 20 for the free Currents tier")

    gnews_page_size = int(os.getenv("GNEWS_PAGE_SIZE", "10"))
    if gnews_page_size < 1 or gnews_page_size > 100:
        raise ValueError("GNEWS_PAGE_SIZE must be between 1 and 100")

    nytimes_page_size = int(os.getenv("NYTIMES_PAGE_SIZE", "10"))
    if nytimes_page_size != 10:
        raise ValueError("NYTIMES_PAGE_SIZE must be 10 because Article Search returns 10 results per page")

    newsdata_page_size = int(os.getenv("NEWSDATA_PAGE_SIZE", "10"))
    if newsdata_page_size < 1 or newsdata_page_size > 50:
        raise ValueError("NEWSDATA_PAGE_SIZE must be between 1 and 50")

    thenewsapi_page_size = int(os.getenv("THENEWSAPI_PAGE_SIZE", "10"))
    if thenewsapi_page_size < 1 or thenewsapi_page_size > 100:
        raise ValueError("THENEWSAPI_PAGE_SIZE must be between 1 and 100")

    language = os.getenv("NEWSAPI_LANGUAGE", "en").strip() or None

    return Settings(
        newsapi_api_key=api_key,
        currents_api_key=currents_api_key,
        gnews_api_key=gnews_api_key,
        nytimes_api_key=nytimes_api_key,
        newsdata_api_key=newsdata_api_key,
        thenewsapi_token=thenewsapi_token,
        database_url=database_url,
        query=query or (queries[0] if queries else ""),
        queries=queries or ((query,) if query else ()),
        max_calls=max_calls,
        page_size=page_size,
        max_pages_per_window=max_pages_per_window,
        currents_page_size=currents_page_size,
        gnews_page_size=gnews_page_size,
        nytimes_page_size=nytimes_page_size,
        newsdata_page_size=newsdata_page_size,
        thenewsapi_page_size=thenewsapi_page_size,
        language=language,
        sort_by=os.getenv("NEWSAPI_SORT_BY", "publishedAt").strip() or "publishedAt",
        timezone=os.getenv("NEWSAPI_TIMEZONE", "Europe/London").strip() or "Europe/London",
    )


def parse_queries(value: str) -> tuple[str, ...]:
    return tuple(query.strip() for query in value.split(";") if query.strip())
