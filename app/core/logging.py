"""Structured (JSON) logging with a per-request trace id bound through contextvars."""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar

import structlog

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)


def new_trace_id() -> str:
    trace_id = uuid.uuid4().hex
    _trace_id.set(trace_id)
    structlog.contextvars.bind_contextvars(trace_id=trace_id)
    return trace_id


def set_trace_id(trace_id: str) -> str:
    _trace_id.set(trace_id)
    structlog.contextvars.bind_contextvars(trace_id=trace_id)
    return trace_id


def current_trace_id() -> str | None:
    return _trace_id.get()


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        timestamper,
    ]
    renderer = (
        structlog.processors.JSONRenderer(ensure_ascii=False)
        if json_logs
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[*shared, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )
    # Quiet noisy third-party loggers.
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
    for noisy in ("httpx", "chromadb", "urllib3", "primp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
