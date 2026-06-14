"""``parser-demo`` console entry point."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from parser_demo.app import DemoConfig, launch


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_args(argv: Sequence[str] | None = None) -> DemoConfig:
    parser = argparse.ArgumentParser(
        prog="parser-demo",
        description="Single-page Gradio demo for omniparser-core.",
    )
    parser.add_argument(
        "--detector-weights",
        type=Path,
        required=True,
        help="Path to YOLO .pt detector weights",
    )
    parser.add_argument(
        "--captioner-weights",
        type=Path,
        required=True,
        help="Directory with the caption model checkpoint",
    )
    parser.add_argument(
        "--captioner-backend",
        choices=("florence2", "blip2", "phi3v"),
        default="florence2",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument(
        "--share",
        action="store_true",
        help="Expose a public gradio.live tunnel (default: off)",
    )
    args = parser.parse_args(argv)

    return DemoConfig(
        detector_weights=args.detector_weights,
        captioner_weights=args.captioner_weights,
        captioner_backend=args.captioner_backend,
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Console entry point — parse args, configure logging, launch demo."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--log-level", default="INFO")
    log_args, rest = pre.parse_known_args(argv)
    _configure_logging(log_args.log_level)

    cfg = _parse_args(rest)
    launch(cfg)


if __name__ == "__main__":
    main()
