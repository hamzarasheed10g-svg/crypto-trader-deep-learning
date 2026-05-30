"""Project-wide logging configuration.

Usage
-----
>>> from utils.logging import get_logger
>>> log = get_logger(__name__)
>>> log.info("ready")
"""
from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def _configure_root(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    # Try rich; fall back to a plain stream handler if unavailable.
    handler: logging.Handler
    try:
        from rich.logging import RichHandler
        handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False, markup=False)
        fmt = "%(message)s"
    except Exception:  # pragma: no cover - fallback path
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S", handlers=[handler])
    _CONFIGURED = True


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """Return a configured logger. ``level`` falls back to LOG_LEVEL env or INFO."""
    _configure_root(level or os.getenv("LOG_LEVEL", "INFO"))
    return logging.getLogger(name)
