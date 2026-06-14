"""Eval driver: iterate dataset, call grounding model, write JSONL, score."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from PIL import Image

from omniparser_eval.data import (
    iter_dataset,
    load_jsonl,
    write_jsonl_record,
)
from omniparser_eval.scoring import compute_correctness, summarize

if TYPE_CHECKING:
    from omniparser_eval.data import PredictionRecord
    from omniparser_eval.models import GroundModel
    from omniparser_eval.scoring import Summary

__all__ = ["run_eval"]

_logger = logging.getLogger(__name__)

GroundingMethod = Literal["positive", "allow_negative", "with_uncertainty"]


def _select_method(model: GroundModel, method: GroundingMethod) -> object:
    """Map method name to the bound method on the model."""
    return {
        "positive": model.ground_only_positive,
        "allow_negative": model.ground_allow_negative,
        "with_uncertainty": model.ground_with_uncertainty,
    }[method]


def _completed_idx_set(output_path: Path) -> set[int]:
    """Read existing JSONL output to support resume-from-last-idx."""
    if not output_path.exists():
        return set()
    return {row["idx"] for row in load_jsonl(output_path) if "idx" in row}


def run_eval(
    *,
    model: GroundModel,
    dataset_root: Path | str,
    output_path: Path | str,
    method: GroundingMethod = "positive",
    max_rows: int | None = None,
    resume: bool = True,
) -> Summary:
    """Run grounding model over a dataset and write JSONL predictions.

    Args:
        model: any object satisfying :class:`omniparser_eval.GroundModel`.
        dataset_root: local path to a downloaded ScreenSpot-Pro snapshot.
        output_path: JSONL file to append predictions to.
        method: which grounding strategy to use.
        max_rows: limit (useful for smoke tests / CI).
        resume: skip rows whose ``idx`` is already in ``output_path``.

    Returns:
        :class:`Summary` of correctness verdicts.
    """
    dataset_root = Path(dataset_root)
    output_path = Path(output_path)
    seen = _completed_idx_set(output_path) if resume else set()
    pairs: list[
        tuple[
            PredictionRecord,
            Literal["correct", "wrong", "negative_correct", "negative_wrong", "failed"],
        ]
    ] = []

    fn = _select_method(model, method)

    processed = 0
    for record in iter_dataset(dataset_root):
        if max_rows is not None and processed >= max_rows:
            break
        idx = record.get("idx", processed)
        if idx in seen:
            continue

        img_full = dataset_root / str(record["img_path"]).lstrip("./")
        try:
            with Image.open(img_full) as raw:
                image = raw.convert("RGB")
                pred = fn(record["prompt_to_evaluate"], image)  # type: ignore[operator]
        except (FileNotFoundError, OSError):
            _logger.exception("image load failed for idx=%s path=%s", idx, img_full)
            verdict = compute_correctness(record, None)
            record["pred"] = None
            record["raw_response"] = None
            record["correctness"] = verdict
            write_jsonl_record(output_path, record)
            pairs.append((record, verdict))
            processed += 1
            continue

        verdict = compute_correctness(record, pred)
        record["pred"] = pred.get("point")
        record["raw_response"] = pred.get("raw_response")
        record["correctness"] = verdict
        write_jsonl_record(output_path, record)
        pairs.append((record, verdict))
        processed += 1

    return summarize(pairs)
