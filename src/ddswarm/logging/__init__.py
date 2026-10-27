"""Structured logging: JSONL event log + content hashing + stdlib log setup."""

from __future__ import annotations

import logging
import os

from .event_writer import EventWriter
from .hashing import hash_content, hash_jsonable

__all__ = ["EventWriter", "hash_content", "hash_jsonable", "get_logger", "configure_logging"]

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def get_logger(name: str) -> logging.Logger:
    """A stdlib logger under the ``ddswarm`` hierarchy (e.g. ``ddswarm.runner``)."""
    return logging.getLogger(f"ddswarm.{name}")


def configure_logging(level: str | int | None = None) -> None:
    """Configure the root logger (idempotent) for CLI/library use.

    The level comes from ``$DDSWARM_LOG_LEVEL`` (default ``WARNING`` — quiet
    unless raised). ``basicConfig`` only takes effect once, so call this at the
    process entry point before any logging happens.
    """
    if level is None:
        level = os.environ.get("DDSWARM_LOG_LEVEL", "WARNING")
    logging.basicConfig(level=level, format=_LOG_FORMAT)
