from __future__ import annotations

import argparse
from dataclasses import asdict

from news_ingest.ingest import fetch_today_yesterday_articles, parse_query_arg
from news_ingest.logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a throttled NewsAPI ingestion with a 100-call budget.")
    parser.add_argument("--query", help="Override NEWSAPI_QUERY for this run.")
    parser.add_argument("--queries", help="Semicolon-separated query list. Overrides NEWSAPI_QUERIES.")
    parser.add_argument("--max-calls", type=int, default=100, help="Maximum NewsAPI page requests to make.")
    parser.add_argument("--sleep-seconds", type=float, default=2, help="Seconds to sleep between NewsAPI requests.")
    parser.add_argument("--last-days", type=int, help="Fetch each day in the last N days, including today.")
    parser.add_argument("--from-date", help="Inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--to-date", help="Inclusive end date in YYYY-MM-DD format.")
    parser.add_argument("--refresh-existing", action="store_true", help="Re-fetch query/date windows that already have stored articles.")
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    if args.max_calls < 1:
        raise ValueError("--max-calls must be at least 1")
    if args.sleep_seconds < 0:
        raise ValueError("--sleep-seconds cannot be negative")

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
