from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterator


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record), "level": record.levelname,
            "logger": record.name, "message": record.getMessage(),
        }
        for key in ("request_id", "job_id", "elapsed_ms", "operation"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(runtime_dir: Path, level: int = logging.INFO) -> None:
    log_dir = runtime_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_dir / "sell-rag.jsonl", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(item, RotatingFileHandler) for item in root.handlers):
        root.addHandler(handler)


@contextmanager
def timed(logger: logging.Logger, operation: str, **context: str) -> Iterator[None]:
    started = time.perf_counter()
    try:
        yield
    finally:
        logger.info(operation, extra={"operation": operation,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2), **context})

