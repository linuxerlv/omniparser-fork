"""Console entry point for the omnitool Gradio UI.

Replaces upstream's ``python app_new.py --windows_host_url ... --omniparser_server_url ...``
with a console script: ``omnitool-ui --windows-host-url ... --omniparser-server-url ...``.

The legacy underscore form is accepted as an alias so operator runbooks
continue to work; only the kebab-case form is canonical.

The underlying ``omnitool_ui.app`` module is a 760-line Gradio script with
top-level argparse and module-level state (``RUN_FOLDER``, ``demo``).
Rather than refactor it into a function (high risk, low immediate value),
this CLI rewrites ``sys.argv`` to the legacy underscore flags and runs the
module via :func:`runpy.run_module`. This preserves byte-level upstream
behavior while still giving us a proper ``project.scripts`` entry point.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnitool-ui",
        description="Launch the OmniTool Gradio UI.",
    )
    parser.add_argument(
        "--windows-host-url",
        "--windows_host_url",
        dest="windows_host_url",
        type=str,
        default="localhost:8006",
        help="omnibox VM host (default: localhost:8006).",
    )
    parser.add_argument(
        "--omniparser-server-url",
        "--omniparser_server_url",
        dest="omniparser_server_url",
        type=str,
        default="localhost:8000",
        help="omniparser-server base URL without scheme (default: localhost:8000).",
    )
    parser.add_argument(
        "--run-folder",
        "--run_folder",
        dest="run_folder",
        type=str,
        default="./tmp/outputs",
        help="Directory for screenshots, SoM images, and run artefacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse args and launch the Gradio app via :func:`runpy.run_module`."""
    args = _build_parser().parse_args(argv)
    sys.argv = [
        "omnitool-ui",
        "--windows_host_url",
        args.windows_host_url,
        "--omniparser_server_url",
        args.omniparser_server_url,
        "--run_folder",
        args.run_folder,
    ]
    runpy.run_module("omnitool_ui.app", run_name="__main__")
