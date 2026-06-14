"""Command-line entry point: ``omniparser-eval``.

This CLI runs the ``allow_negative`` and ``with_uncertainty`` grounding
methods, which take raw screenshots and don't need an Omniparser
instance. The ``positive`` method requires a configured parser with YOLO
+ caption + OCR weights loaded; running it from the CLI requires
programmatic configuration via :func:`omniparser_eval.run_eval` rather
than this entry point.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from omniparser_eval.data import download_screenspot_pro
from omniparser_eval.driver import run_eval

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="omniparser-eval",
        description="Run grounding benchmarks (ScreenSpot-Pro) for OmniParser.",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        help="Local ScreenSpot-Pro snapshot. If omitted, downloads via huggingface_hub.",
    )
    p.add_argument(
        "--mirror",
        default=None,
        help="HF endpoint: 'hf', 'hf-mirror', or a URL. Default: $HF_ENDPOINT or hf-mirror.com.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSONL output path (append-mode).",
    )
    p.add_argument(
        "--method",
        choices=("allow_negative", "with_uncertainty"),
        default="allow_negative",
        help=(
            "Grounding strategy. The 'positive' method needs a configured Omniparser parser "
            "and is only available via the Python API (omniparser_eval.run_eval)."
        ),
    )
    p.add_argument(
        "--model-name",
        default="gpt-4o-2024-05-13",
        help="OpenAI model name.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit dataset rows (useful for smoke tests).",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-evaluate rows already present in the output JSONL.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v INFO, -vv DEBUG).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns process exit code."""
    args = _build_parser().parse_args(argv)
    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Lazy imports so `omniparser-eval --help` doesn't pull in openai.
    import openai

    from omniparser_eval.models import OpenAIGroundModel

    if not os.environ.get("OPENAI_API_KEY"):
        sys.stderr.write("error: OPENAI_API_KEY is not set\n")
        return 2

    dataset_root = args.dataset_root
    if dataset_root is None:
        dataset_root = download_screenspot_pro(mirror=args.mirror)

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = OpenAIGroundModel(client=client, parser=None, model_name=args.model_name)

    summary = run_eval(
        model=model,
        dataset_root=dataset_root,
        output_path=args.output,
        method=args.method,
        max_rows=args.max_rows,
        resume=not args.no_resume,
    )

    sys.stdout.write(
        f"total={summary.total} correct={summary.correct} wrong={summary.wrong} "
        f"negative_correct={summary.negative_correct} negative_wrong={summary.negative_wrong} "
        f"failed={summary.failed} accuracy={summary.overall:.4f}\n",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
