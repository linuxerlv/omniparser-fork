"""``omniparser-server`` console entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import uvicorn

from omniparser_server.logging_config import configure_logging, get_logger
from omniparser_server.settings import ServerSettings


def _build_settings(argv: Sequence[str] | None = None) -> ServerSettings:
    parser = argparse.ArgumentParser(
        prog="omniparser-server",
        description="FastAPI inference service wrapping omniparser-core.",
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-format", default=None, choices=["json", "console"])
    parser.add_argument("--eager-load", action="store_true", default=None)
    parser.add_argument("--print-settings", action="store_true")
    args = parser.parse_args(argv)

    overrides: dict[str, object] = {}
    if args.host is not None:
        overrides["host"] = args.host
    if args.port is not None:
        overrides["port"] = args.port
    if args.log_level is not None:
        overrides["log_level"] = args.log_level
    if args.log_format is not None:
        overrides["log_format"] = args.log_format
    if args.eager_load is not None:
        overrides["eager_load"] = args.eager_load

    settings = ServerSettings(**overrides)  # type: ignore[arg-type]
    if args.print_settings:
        sys.stdout.write(settings.model_dump_json(indent=2) + "\n")
        raise SystemExit(0)
    return settings


def main(argv: Sequence[str] | None = None) -> None:
    """Console entry point — boot uvicorn with configured settings."""
    settings = _build_settings(argv)
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    log = get_logger(__name__)
    log.info(
        "starting_uvicorn",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )
    uvicorn.run(
        "omniparser_server.app:create_app",
        host=settings.host,
        port=settings.port,
        log_config=None,
        factory=True,
        reload=False,
    )


if __name__ == "__main__":
    main()
