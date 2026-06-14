"""Pure geometric utilities for axis-aligned bounding boxes.

All boxes are 4-tuples ``(x1, y1, x2, y2)`` in any consistent coordinate
system (pixel space or normalised [0,1] - the helpers do not care, as
long as both inputs use the same scale).

This module deliberately depends on nothing but the standard library so
that callers can import it without paying for torch / opencv / ultralytics.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "Box",
    "box_area",
    "int_box_area",
    "intersection_area",
    "iou",
    "is_inside",
]

# Axis-aligned box: (x1, y1, x2, y2). Kept as a structural alias so it
# accepts tuples, lists, and torch.Tensor row slices alike.
Box = Sequence[float]


def box_area(box: Box) -> float:
    """Area of an axis-aligned box in input coordinates."""
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection_area(box1: Box, box2: Box) -> float:
    """Area of the intersection rectangle of two boxes (0 if disjoint)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(box1: Box, box2: Box, *, return_max: bool = True) -> float:
    """Generalised IoU between two boxes.

    By default returns ``max(IoU, ratio_of_box1, ratio_of_box2)`` to match
    upstream OmniParser's overlap criterion (it treats fully-contained
    boxes as ``1.0`` even if their union is much larger). Pass
    ``return_max=False`` to get the standard IoU.

    Adds a tiny epsilon to the union to avoid division by zero on
    degenerate zero-area inputs.
    """
    inter = intersection_area(box1, box2)
    a1 = box_area(box1)
    a2 = box_area(box2)
    union = a1 + a2 - inter + 1e-6
    if a1 > 0 and a2 > 0:
        ratio1 = inter / a1
        ratio2 = inter / a2
    else:
        ratio1 = 0.0
        ratio2 = 0.0
    standard_iou = inter / union
    return max(standard_iou, ratio1, ratio2) if return_max else standard_iou


def is_inside(box1: Box, box2: Box, *, threshold: float = 0.80) -> bool:
    """True if ``box1`` is mostly inside ``box2``.

    Uses ``intersection / area(box1)`` rather than strict coordinate
    containment to tolerate small annotation noise. The default 0.80
    threshold matches the icon-vs-OCR overlap test used by
    ``remove_overlap_new`` in upstream OmniParser.
    """
    a1 = box_area(box1)
    if a1 <= 0:
        return False
    return intersection_area(box1, box2) / a1 > threshold


def int_box_area(box: Box, width: int, height: int) -> int:
    """Pixel area of a normalised ``(x1, y1, x2, y2)`` box.

    Inputs are assumed to be in [0, 1]. Used to drop zero-area boxes
    after the YOLO output is rescaled by image size.
    """
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    ix1, iy1, ix2, iy2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
    return (ix2 - ix1) * (iy2 - iy1)
