from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict
from datetime import UTC, datetime

from news_ingest.config import Settings, load_settings
from news_ingest.db import connect, ensure_schema, stored_article_count, upsert_article
from news_ingest.ingest import IngestSummary, _effective_queries, build_date_windows, parse_query_arg
from news_ingest.logging_config import configure_logging
from news_ingest.newsdata_client import NewsDataClient, NewsDataRequestError, newsdata_to_article


logger = logging.getLogger(__name__)


def fetch_newsdata_articles(
    *,
    settings: Settings | None = None,
    query: str | None = None,
    queries: tuple[str, ...] | None = None,
    max_calls: int | None = None,
    sleep_seconds: float = 0,
    last_days: int | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    refresh_existing: bool = False,
    latest_only: bool = False,
) -> IngestSummary:
    configure_logging()
    settings = settings or load_settings(require_newsapi=False, require_newsdata=True)
    if not settings.newsdata_api_key:
        raise ValueError("Missing required environment variable: NEWSDATA_API_KEY")

    effective_queries = _effective_queries(settings=settings, query=query, queries=queries)
    effective_max_calls = max_calls or min(settings.max_calls, 100)
    windows = [None] if latest_only else build_date_windows(settings=settings, last_days=last_days, from_date=from_date, to_date=to_date)
    client = NewsDataClient(settings.newsdata_api_key)
    summary = IngestSummary()

    logger.info(
        "Starting NewsData ingestion queries=%s windows=%s max_calls=%s sleep_seconds=%s refresh_existing=%s",
        effective_queries,
        [window.label if window is not None else "latest" for window in windows],
        effective_max_calls,
        sleep_seconds,
        refresh_existing,
    )

    with connect(settings.database_url) as conn:
        ensure_schema(conn)
        for effective_query in effective_queries:
            for window in windows:
                window_label = window.label if window is not None else "latest"
                if not refresh_existing:
                    existing_count = stored_article_count(conn, query=effective_query, window_label=window_label, provider="newsdata")
                    if existing_count > 0:
                        summary.skipped_windows += 1
                        logger.info(
                            "Skipping already-fetched NewsData window query=%r window=%s existing_articles=%s",
                            effective_query,
                            window_label,
                            existing_count,
                        )
                        continue

                _ingest_newsdata_window(
                    conn=conn,
                    client=client,
                    settings=settings,
                    query=effective_query,
                    window=window,
                    summary=summary,
                    max_calls=effective_max_calls,
                    sleep_seconds=sleep_seconds,
                    latest_only=latest_only,
                )
                if summary.calls_used >= effective_max_calls or summary.stopped_reason:
                    break
            if summary.calls_used >= effective_max_calls or summary.stopped_reason:
                break

    logger.info("Finished NewsData ingestion summary=%s", asdict(summary))
    return summary


def _ingest_newsdata_window(*, conn, client: NewsDataClient, settings: Settings, query, window, summary, max_calls, sleep_seconds, latest_only: bool = False) -> None:
    page_token: str | None = None
    page_count = 0
    page_size = settings.newsdata_page_size
    while summary.calls_used < max_calls:
        if summary.calls_used > 0 and sleep_seconds > 0:
            time.sleep(sleep_seconds)

        window_label = window.label if window is not None else "latest"
        logger.info("Requesting NewsData page query=%r window=%s page_token=%s size=%s", query, window_label, page_token, page_size)
        try:
            if latest_only:
                api_page = client.latest_page(
                    query=query,
                    language=settings.language,
                    page_size=page_size,
                    page=page_token,
                )
            else:
                api_page = client.archive_page(
                    query=query,
                    language=settings.language,
                    page_size=page_size,
                    page=page_token,
                    from_date=window.label,
                    to_date=window.label,
                )
        except NewsDataRequestError as exc:
            if _is_newsdata_rate_limited(exc):
                summary.stopped_reason = "rateLimited"
                logger.warning(
                    "Stopping NewsData ingestion reason=quota_or_rate_limit query=%r window=%s code=%s message=%s",
                    query,
                    window_label,
                    exc.code,
                    exc.message,
                )
            else:
                summary.errors += 1
                summary.stopped_reason = f"newsdata_{exc.code}"
                logger.error(
                    "Stopping NewsData ingestion reason=request_failed query=%r window=%s code=%s message=%s",
                    query,
                    window_label,
                    exc.code,
                    exc.message,
                )
            break
        except Exception:
            summary.errors += 1
            summary.stopped_reason = "newsdata_unexpected_error"
            logger.exception("Unexpected NewsData request failure query=%r window=%s", query, window_label)
            break

        summary.calls_used += 1
        page_count += 1
        summary.articles_received += len(api_page.articles)
        logger.info(
            "Received NewsData page query=%r window=%s articles=%s total_results=%s next_page=%s",
            query,
            window_label,
            len(api_page.articles),
            api_page.total_results,
            bool(api_page.next_page),
        )

        fetched_at = datetime.now(UTC)
        for raw_article in api_page.articles:
            article = newsdata_to_article(raw_article)
            try:
                result = upsert_article(
                    conn,
                    article,
                    query=query,
                    window_label=window_label,
                    api_page=page_count,
                    fetched_at=fetched_at,
                    provider="newsdata",
                )
                if result is None or getattr(result, "duplicate_content", False):
                    summary.skipped += 1
                elif result.inserted:
                    summary.inserted += 1
                else:
                    summary.updated += 1
            except Exception:
                conn.rollback()
                summary.errors += 1
                logger.exception("Failed to upsert NewsData article url=%r", article.get("url"))
            else:
                conn.commit()

        if not api_page.next_page:
            logger.info("Stopping NewsData window query=%r window=%s reason=no_next_page", query, window_label)
            break
        if page_count >= getattr(settings, "max_pages_per_window", 1):
            logger.warning(
                "Stopping NewsData window query=%r window=%s reason=max_pages_per_window pages=%s limit=%s",
                query,
                window_label,
                page_count,
                getattr(settings, "max_pages_per_window", 1),
            )
            break
        page_token = api_page.next_page


def _is_newsdata_rate_limited(exc: NewsDataRequestError) -> bool:
    value = f"{exc.code} {exc.message}".lower()
    return "rate" in value or "quota" in value or "limit" in value or exc.code in {"402", "403", "429"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch NewsData.io articles into PostgreSQL.")
    parser.add_argument("--query", help="Override NEWSAPI_QUERY for this run.")
    parser.add_argument("--queries", help="Semicolon-separated query list. Overrides NEWSAPI_QUERIES.")
    parser.add_argument("--max-calls", type=int, default=100, help="Maximum NewsData.io requests for this run.")
    parser.add_argument("--sleep-seconds", type=float, default=0, help="Seconds to sleep between NewsData.io requests.")
    parser.add_argument("--last-days", type=int, help="Fetch each day in the last N days, including today.")
    parser.add_argument("--from-date", help="Inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--to-date", help="Inclusive end date in YYYY-MM-DD format.")
    parser.add_argument("--refresh-existing", action="store_true", help="Re-fetch NewsData query/date windows that already have stored articles.")
    parser.add_argument("--latest-only", action="store_true", help="Use /latest without date filters. Use this on plans that do not include /archive access.")
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    summary = fetch_newsdata_articles(
        query=args.query,
        queries=parse_query_arg(args.queries),
        max_calls=args.max_calls,
        sleep_seconds=args.sleep_seconds,
        last_days=args.last_days,
        from_date=args.from_date,
        to_date=args.to_date,
        refresh_existing=args.refresh_existing,
        latest_only=args.latest_only,
    )
    print(asdict(summary))


if __name__ == "__main__":
    main()
