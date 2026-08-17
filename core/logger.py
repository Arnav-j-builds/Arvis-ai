"""
core.logger
~~~~~~~~~~~

A thin wrapper over :mod:`logging` that gives every module a consistent
prefix. The level can be overridden via the ``JARVIS_LOG_LEVEL`` environment
variable (default: ``INFO``).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

_DEFAULT_LEVEL: Final[str] = "INFO"
_LOG_FORMAT: Final[str] = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

_configured = False


def configure_logging(level: str | int | None = None) -> None:
    """Configure the root logger exactly once.

    Parameters
    ----------
    level:
        A log level name (``"DEBUG"`` ...) or numeric level. If *None* we
        honour ``JARVIS_LOG_LEVEL`` from the environment, falling back to
        :data:`_DEFAULT_LEVEL`.
    """
    global _configured
    if _configured:
        return

    if level is None:
        level = os.getenv("JARVIS_LOG_LEVEL", _DEFAULT_LEVEL)

    if isinstance(level, str):
        level = level.upper()
        level = getattr(logging, level, logging.INFO)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Tame noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the standard arvis configuration."""
    configure_logging()
    return logging.getLogger(name)
