from __future__ import annotations

import logging
from pathlib import Path


LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ExactLevelFilter(logging.Filter):
    def __init__(self, level: int) -> None:
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == self.level


def configure_logging(log_dir: str | Path = "logs") -> None:
    root = logging.getLogger()
    if getattr(root, "_news_rag_logging_configured", False):
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    root.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    for level, filename in (
        (logging.INFO, "info.log"),
        (logging.WARNING, "warning.log"),
        (logging.ERROR, "error.log"),
    ):
        handler = logging.FileHandler(log_path / filename, encoding="utf-8")
        handler.setLevel(level)
        handler.addFilter(ExactLevelFilter(level))
        handler.setFormatter(formatter)
        root.addHandler(handler)

    root._news_rag_logging_configured = True
