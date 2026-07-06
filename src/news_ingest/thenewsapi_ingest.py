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
from news_ingest.thenewsapi_client import TheNewsApiClient, TheNewsApiRequestError, thenewsapi_to_article


logger = logging.getLogger(__name__)


def fetch_thenewsapi_articles(
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
) -> IngestSummary:
    configure_logging()
    settings = settings or load_settings(require_newsapi=False, require_thenewsapi=True)
    if not settings.thenewsapi_token:
        raise ValueError("Missing required environment variable: THENEWSAPI_TOKEN")

    effective_queries = _effective_queries(settings=settings, query=query, queries=queries)
    effective_max_calls = max_calls or min(settings.max_calls, 100)
    windows = build_date_windows(settings=settings, last_days=last_days, from_date=from_date, to_date=to_date)
    client = TheNewsApiClient(settings.thenewsapi_token)
    summary = IngestSummary()

    logger.info(
        "Starting TheNewsAPI ingestion queries=%s windows=%s max_calls=%s sleep_seconds=%s refresh_existing=%s",
        effective_queries,
        [window.label for window in windows],
        effective_max_calls,
        sleep_seconds,
        refresh_existing,
    )

    with connect(settings.database_url) as conn:
        ensure_schema(conn)
        for effective_query in effective_queries:
            for window in windows:
                if not refresh_existing:
                    existing_count = stored_article_count(conn, query=effective_query, window_label=window.label, provider="thenewsapi")
                    if existing_count > 0:
                        summary.skipped_windows += 1
                        logger.info(
                            "Skipping already-fetched TheNewsAPI window query=%r window=%s existing_articles=%s",
                            effective_query,
                            window.label,
                            existing_count,
                        )
                        continue

                _ingest_thenewsapi_window(
                    conn=conn,
                    client=client,
                    settings=settings,
                    query=effective_query,
                    window=window,
                    summary=summary,
                    max_calls=effective_max_calls,
                    sleep_seconds=sleep_seconds,
                )
                if summary.calls_used >= effective_max_calls or summary.stopped_reason:
                    break
            if summary.calls_used >= effective_max_calls or summary.stopped_reason:
                break

    logger.info("Finished TheNewsAPI ingestion summary=%s", asdict(summary))
    return summary


def _ingest_thenewsapi_window(*, conn, client: TheNewsApiClient, settings: Settings, query, window, summary, max_calls, sleep_seconds) -> None:
    page = 1
    page_size = settings.thenewsapi_page_size
    while summary.calls_used < max_calls:
        if summary.calls_used > 0 and sleep_seconds > 0:
            time.sleep(sleep_seconds)

        logger.info("Requesting TheNewsAPI page query=%r window=%s page=%s limit=%s", query, window.label, page, page_size)
        try:
            api_page = client.all_news_page(
                query=query,
                language=settings.language,
                page_size=page_size,
                page=page,
                published_on=window.label,
            )
        except TheNewsApiRequestError as exc:
            if _is_thenewsapi_rate_limited(exc):
                summary.stopped_reason = "rateLimited"
                logger.warning(
                    "Stopping TheNewsAPI ingestion reason=quota_or_rate_limit query=%r window=%s page=%s code=%s message=%s",
                    query,
                    window.label,
                    page,
                    exc.code,
                    exc.message,
                )
            else:
                summary.errors += 1
                summary.stopped_reason = f"thenewsapi_{exc.code}"
                logger.error(
                    "Stopping TheNewsAPI ingestion reason=request_failed query=%r window=%s page=%s code=%s message=%s",
                    query,
                    window.label,
                    page,
                    exc.code,
                    exc.message,
                )
            break
        except Exception:
            summary.errors += 1
            summary.stopped_reason = "thenewsapi_unexpected_error"
            logger.exception("Unexpected TheNewsAPI request failure query=%r window=%s page=%s", query, window.label, page)
            break

        summary.calls_used += 1
        summary.articles_received += len(api_page.articles)
        logger.info(
            "Received TheNewsAPI page query=%r window=%s page=%s articles=%s found=%s returned=%s",
            query,
            window.label,
            page,
            len(api_page.articles),
            api_page.found,
            api_page.returned,
        )

        fetched_at = datetime.now(UTC)
        for raw_article in api_page.articles:
            article = thenewsapi_to_article(raw_article)
            try:
                result = upsert_article(
                    conn,
                    article,
                    query=query,
                    window_label=window.label,
                    api_page=page,
                    fetched_at=fetched_at,
                    provider="thenewsapi",
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
                logger.exception("Failed to upsert TheNewsAPI article url=%r", article.get("url"))
            else:
                conn.commit()

        if len(api_page.articles) < page_size or api_page.returned < page_size:
            logger.info(
                "Stopping TheNewsAPI window query=%r window=%s reason=short_page articles=%s returned=%s",
                query,
                window.label,
                len(api_page.articles),
                api_page.returned,
            )
            break
        if api_page.found > 0 and page * page_size >= api_page.found:
            logger.info("Stopping TheNewsAPI window query=%r window=%s reason=all_results_seen page=%s found=%s", query, window.label, page, api_page.found)
            break
        if page >= getattr(settings, "max_pages_per_window", 1):
            logger.warning(
                "Stopping TheNewsAPI window query=%r window=%s reason=max_pages_per_window page=%s limit=%s",
                query,
                window.label,
                page,
                getattr(settings, "max_pages_per_window", 1),
            )
            break
        page += 1


def _is_thenewsapi_rate_limited(exc: TheNewsApiRequestError) -> bool:
    value = f"{exc.code} {exc.message}".lower()
    return "rate" in value or "quota" in value or "limit" in value or exc.code in {"402", "403", "429"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch TheNewsAPI articles into PostgreSQL.")
    parser.add_argument("--query", help="Override NEWSAPI_QUERY for this run.")
    parser.add_argument("--queries", help="Semicolon-separated query list. Overrides NEWSAPI_QUERIES.")
    parser.add_argument("--max-calls", type=int, default=100, help="Maximum TheNewsAPI requests for this run.")
    parser.add_argument("--sleep-seconds", type=float, default=0, help="Seconds to sleep between TheNewsAPI requests.")
    parser.add_argument("--last-days", type=int, help="Fetch each day in the last N days, including today.")
    parser.add_argument("--from-date", help="Inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--to-date", help="Inclusive end date in YYYY-MM-DD format.")
    parser.add_argument("--refresh-existing", action="store_true", help="Re-fetch TheNewsAPI query/date windows that already have stored articles.")
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    summary = fetch_thenewsapi_articles(
        query=args.query,
        queries=parse_query_arg(args.queries),
        max_calls=args.max_calls,
        sleep_seconds=args.sleep_seconds,
        last_days=args.last_days,
        from_date=args.from_date,
        to_date=args.to_date,
        refresh_existing=args.refresh_existing,
    )
    print(asdict(summary))


if __name__ == "__main__":
    main()
