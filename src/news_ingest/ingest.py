from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from collections.abc import Sequence

from news_ingest.config import Settings, load_settings
from news_ingest.dates import DateWindow, date_range_windows, last_n_days_windows, parse_date, today_yesterday_windows
from news_ingest.db import connect, ensure_schema, stored_article_count, upsert_article
from news_ingest.logging_config import configure_logging
from news_ingest.newsapi_client import NewsApiEverythingClient, NewsApiRequestError


logger = logging.getLogger(__name__)


@dataclass
class IngestSummary:
    calls_used: int = 0
    articles_received: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    errors: int = 0
    stopped_reason: str | None = None
    skipped_windows: int = 0


def fetch_today_yesterday_articles(
    *,
    settings: Settings | None = None,
    query: str | None = None,
    queries: Sequence[str] | None = None,
    max_calls: int | None = None,
    sleep_seconds: float = 0,
    last_days: int | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    refresh_existing: bool = False,
) -> IngestSummary:
    configure_logging()
    settings = settings or load_settings()
    effective_queries = _effective_queries(settings=settings, query=query, queries=queries)
    effective_max_calls = max_calls or settings.max_calls

    client = NewsApiEverythingClient(settings.newsapi_api_key)
    windows = build_date_windows(settings=settings, last_days=last_days, from_date=from_date, to_date=to_date)
    summary = IngestSummary()

    logger.info(
        "Starting NewsAPI ingestion queries=%s windows=%s max_calls=%s sleep_seconds=%s refresh_existing=%s",
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
                    existing_count = stored_article_count(conn, query=effective_query, window_label=window.label)
                    if existing_count > 0:
                        summary.skipped_windows += 1
                        logger.info(
                            "Skipping already-fetched window query=%r window=%s existing_articles=%s",
                            effective_query,
                            window.label,
                            existing_count,
                        )
                        continue

                _ingest_window(
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

    logger.info("Finished NewsAPI ingestion summary=%s", asdict(summary))
    return summary


def build_date_windows(
    *,
    settings: Settings,
    last_days: int | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> list[DateWindow]:
    if last_days is not None and (from_date or to_date):
        raise ValueError("Use either --last-days or --from-date/--to-date, not both")
    if last_days is not None:
        return last_n_days_windows(last_days, tz=settings.zoneinfo)
    if from_date or to_date:
        if not from_date or not to_date:
            raise ValueError("Both --from-date and --to-date are required for a custom range")
        return date_range_windows(parse_date(from_date), parse_date(to_date), tz=settings.zoneinfo)
    return today_yesterday_windows(tz=settings.zoneinfo)


def _effective_queries(*, settings: Settings, query: str | None, queries: Sequence[str] | None) -> tuple[str, ...]:
    if queries:
        values = tuple(value.strip() for value in queries if value.strip())
    elif query:
        values = (query.strip(),)
    else:
        values = settings.queries
    if not values:
        raise ValueError("At least one NewsAPI query is required")
    return values


def _ingest_window(
    *,
    conn,
    client: NewsApiEverythingClient,
    settings: Settings,
    query: str,
    window: DateWindow,
    summary: IngestSummary,
    max_calls: int,
    sleep_seconds: float = 0,
) -> None:
    page = 1
    while summary.calls_used < max_calls:
        if summary.calls_used > 0 and sleep_seconds > 0:
            time.sleep(sleep_seconds)

        logger.info(
            "Requesting NewsAPI page query=%r window=%s page=%s page_size=%s",
            query,
            window.label,
            page,
            settings.page_size,
        )
        try:
            api_page = client.get_everything_page(
                query=query,
                from_iso=window.from_iso,
                to_iso=window.to_iso,
                page_size=settings.page_size,
                page=page,
                language=settings.language,
                sort_by=settings.sort_by,
            )
        except NewsApiRequestError as exc:
            if exc.code == "rateLimited":
                summary.stopped_reason = "rateLimited"
                logger.warning(
                    "Stopping ingestion reason=rateLimited query=%r window=%s page=%s message=%s",
                    query,
                    window.label,
                    page,
                    exc.message,
                )
            else:
                summary.errors += 1
                logger.error(
                    "NewsAPI request failed query=%r window=%s page=%s code=%s message=%s",
                    query,
                    window.label,
                    page,
                    exc.code,
                    exc.message,
                )
            break
        except Exception:
            summary.errors += 1
            logger.exception("Unexpected NewsAPI request failure query=%r window=%s page=%s", query, window.label, page)
            break

        summary.calls_used += 1
        summary.articles_received += len(api_page.articles)
        logger.info(
            "Received NewsAPI page query=%r window=%s page=%s articles=%s total_results=%s",
            query,
            window.label,
            page,
            len(api_page.articles),
            api_page.total_results,
        )

        fetched_at = datetime.now(UTC)
        for article in api_page.articles:
            try:
                result = upsert_article(
                    conn,
                    article,
                    query=query,
                    window_label=window.label,
                    api_page=page,
                    fetched_at=fetched_at,
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
                logger.exception("Failed to upsert article url=%r", article.get("url"))
            else:
                conn.commit()

        if len(api_page.articles) < settings.page_size:
            logger.info("Stopping window query=%r window=%s reason=short_page articles=%s", query, window.label, len(api_page.articles))
            break
        if page * settings.page_size >= api_page.total_results:
            logger.info("Stopping window query=%r window=%s reason=all_results_seen page=%s total_results=%s", query, window.label, page, api_page.total_results)
            break
        if page >= getattr(settings, "max_pages_per_window", 1):
            logger.warning(
                "Stopping window query=%r window=%s reason=max_pages_per_window page=%s limit=%s. Free NewsAPI developer accounts cannot request page 2+.",
                query,
                window.label,
                page,
                getattr(settings, "max_pages_per_window", 1),
            )
            break
        if summary.calls_used >= max_calls:
            logger.warning("Stopping ingestion reason=max_calls calls_used=%s max_calls=%s", summary.calls_used, max_calls)
            break
        page += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch NewsAPI articles into PostgreSQL.")
    parser.add_argument("--query", help="Override NEWSAPI_QUERY for this run.")
    parser.add_argument("--queries", help="Semicolon-separated query list. Overrides NEWSAPI_QUERIES.")
    parser.add_argument("--max-calls", type=int, help="Override NEWSAPI_MAX_CALLS for this run.")
    parser.add_argument("--sleep-seconds", type=float, default=0, help="Seconds to sleep between NewsAPI page requests.")
    parser.add_argument("--last-days", type=int, help="Fetch each day in the last N days, including today.")
    parser.add_argument("--from-date", help="Inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--to-date", help="Inclusive end date in YYYY-MM-DD format.")
    parser.add_argument("--refresh-existing", action="store_true", help="Re-fetch query/date windows that already have stored articles.")
    return parser


def parse_query_arg(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(query.strip() for query in value.split(";") if query.strip())


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    summary = fetch_today_yesterday_articles(
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
