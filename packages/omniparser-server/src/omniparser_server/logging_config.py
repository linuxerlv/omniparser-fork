"""structlog configuration for ``omniparser-server``.

Configures structlog with two output formats:

* ``json`` — production-ready JSON lines on stdout.
* ``console`` — colored, human-readable for local development.

Standard library ``logging`` is wired through structlog so libraries that emit
through ``logging`` (uvicorn, fastapi, hf_hub) appear in the same stream with
identical structure.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Final, Literal

import structlog

if TYPE_CHECKING:
    from structlog.types import Processor

LogFormatLiteral = Literal["json", "console"]

_SHARED_PROCESSORS: Final[tuple[Processor, ...]] = (
    structlog.contextvars.merge_contextvars,
    structlog.processors.add_log_level,
    structlog.processors.StackInfoRenderer(),
    structlog.dev.set_exc_info,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
)


def configure_logging(
    *,
    level: str = "INFO",
    fmt: LogFormatLiteral = "json",
) -> None:
    """Configure structlog and the stdlib root logger.

    Idempotent — safe to call multiple times (last call wins). Tests should
    call it in a fixture, not at import time.
    """
    renderer: Processor
    renderer = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[*_SHARED_PROCESSORS, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level],
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(message)s"),
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    for noisy in ("uvicorn.access",):
        logging.getLogger(noisy).setLevel(max(root.level, logging.WARNING))


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger; thin wrapper for type clarity."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
