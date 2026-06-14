"""Correctness scoring and aggregation.

Scoring rules (matched against ``upstream/eval/logs_sspro_omniv2.json``):

* ``gt_type="positive"`` + model returned a ``point``: correct iff the
  predicted point lies inside the GT bbox (closed interval). Otherwise
  ``wrong``.
* ``gt_type="positive"`` + model returned no point: ``wrong``.
* ``gt_type="negative"`` + model returned ``result="negative"``:
  ``negative_correct``, otherwise ``negative_wrong``.
* API/parse failure: ``failed``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable

    from omniparser_eval.data import PredictionRecord
    from omniparser_eval.models import GroundResult

__all__ = [
    "Correctness",
    "Summary",
    "compute_correctness",
    "point_in_bbox",
    "summarize",
]

Correctness = Literal["correct", "wrong", "negative_correct", "negative_wrong", "failed"]


def point_in_bbox(point: list[int] | None, bbox: list[int] | None) -> bool:
    """Return True iff ``point=(x,y)`` is inside ``bbox=(x1,y1,x2,y2)`` (inclusive)."""
    if point is None or bbox is None:
        return False
    if len(point) < 2 or len(bbox) < 4:
        return False
    px, py = point[0], point[1]
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    return x1 <= px <= x2 and y1 <= py <= y2


def compute_correctness(  # noqa: PLR0911 - one return per disjoint scoring case is clearer than nesting
    record: PredictionRecord,
    pred: GroundResult | None,
) -> Correctness:
    """Compute correctness for one (record, prediction) pair.

    Args:
        record: dataset row with ``gt_type`` and ``bbox`` fields.
        pred: model output, or ``None`` if the call failed before
            producing any output.
    """
    gt_type = record.get("gt_type", "positive")
    if pred is None:
        return "failed"
    result = pred.get("result")
    if result == "failed":
        return "failed"

    if gt_type == "positive":
        if result == "negative":
            return "wrong"
        if point_in_bbox(pred.get("point"), record.get("bbox")):
            return "correct"
        return "wrong"

    # gt_type == "negative"
    if result == "negative":
        return "negative_correct"
    return "negative_wrong"


@dataclass(slots=True)
class Summary:
    """Aggregate accuracy with optional per-key breakdown.

    ``overall`` is the fraction of ``correct`` + ``negative_correct`` over
    total non-``failed`` rows. ``by_key`` provides the same fraction
    grouped by an arbitrary record key (e.g. ``"group"``, ``"ui_type"``).
    """

    total: int = 0
    correct: int = 0
    wrong: int = 0
    negative_correct: int = 0
    negative_wrong: int = 0
    failed: int = 0
    by_key: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def scored(self) -> int:
        """Number of rows that produced a non-failed verdict."""
        return self.total - self.failed

    @property
    def overall(self) -> float:
        """Accuracy = (correct + negative_correct) / scored, or 0 if no scored rows."""
        if self.scored == 0:
            return 0.0
        return (self.correct + self.negative_correct) / self.scored


def summarize(
    pairs: Iterable[tuple[PredictionRecord, Correctness]],
    *,
    breakdown_keys: tuple[str, ...] = ("group", "platform", "ui_type"),
) -> Summary:
    """Aggregate correctness verdicts into a :class:`Summary`."""
    summary = Summary()
    bucket: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "wrong": 0, "negative_correct": 0, "negative_wrong": 0, "failed": 0},
    )

    for record, verdict in pairs:
        summary.total += 1
        # Mypy doesn't allow dynamic attr setting on slotted dataclass; use setattr.
        current = getattr(summary, verdict)
        setattr(summary, verdict, current + 1)

        for key in breakdown_keys:
            value = record.get(key)
            if value is None:
                continue
            bucket_key = f"{key}={value}"
            bucket[bucket_key][verdict] += 1

    summary.by_key = dict(bucket)
    return summary
