"""Bounding-box annotation primitives.

Replaces ``upstream/util/box_annotator.py``. The implementation is the
same algorithm — draw a rectangle, draw a label background, place the
label text where it least overlaps existing detections — but cleaned up:

* type annotations on every function
* ``opencv-python-headless`` (no GUI surface dependency)
* ``logging`` for debug breadcrumbs (was ``# import pdb; pdb.set_trace()``)
* IoU helpers reused from :mod:`omniparser.geometry` (deduplicated with
  the inline copy in upstream's ``utils.py``)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import cv2

from omniparser.geometry import iou

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

__all__ = ["BoxAnnotator", "annotate", "get_optimal_label_pos"]

_logger = logging.getLogger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Colour palette borrowed from supervision.draw.color.ColorPalette.DEFAULT
# but inlined here to avoid making `supervision` a hard dependency of
# annotation. Each entry is ``(R, G, B)``.
_DEFAULT_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0xA3, 0x51, 0xFB),
    (0xE6, 0x19, 0x4B),
    (0x3C, 0xB4, 0x4B),
    (0xFF, 0xE1, 0x19),
    (0x07, 0x82, 0xFA),
    (0xF5, 0x82, 0x31),
    (0x91, 0x1E, 0xB4),
    (0x46, 0xF0, 0xF0),
    (0xF0, 0x32, 0xE6),
    (0xBC, 0xF6, 0x0C),
)


def _palette_color(idx: int) -> tuple[int, int, int]:
    return _DEFAULT_PALETTE[idx % len(_DEFAULT_PALETTE)]


class BoxAnnotator:
    """Draw bounding boxes and labels onto a numpy image.

    Mirrors the upstream API surface so that ``parse_screen`` can swap
    between the two without code changes; the relevant bits of
    ``supervision.Detections`` are accessed via attribute access on a
    duck-typed object (``.xyxy`` is required, ``.class_id`` optional).
    """

    def __init__(
        self,
        *,
        thickness: int = 3,
        text_scale: float = 0.5,
        text_thickness: int = 2,
        text_padding: int = 10,
        avoid_overlap: bool = True,
    ) -> None:
        """Initialise drawing parameters; defaults match upstream demo settings."""
        self.thickness = thickness
        self.text_scale = text_scale
        self.text_thickness = text_thickness
        self.text_padding = text_padding
        self.avoid_overlap = avoid_overlap

    def annotate(
        self,
        scene: np.ndarray,
        detections: Any,
        *,
        labels: Sequence[str] | None = None,
        skip_label: bool = False,
        image_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Draw boxes (and optionally labels) for ``detections`` onto ``scene``.

        ``scene`` is mutated in place and also returned. ``detections`` is
        any object with an ``.xyxy`` attribute of shape ``(N, 4)``; an
        optional ``.class_id`` array selects palette entries.
        """
        n = len(detections)
        for i in range(n):
            x1, y1, x2, y2 = detections.xyxy[i].astype(int)
            class_id = (
                detections.class_id[i]
                if getattr(detections, "class_id", None) is not None
                else None
            )
            idx = class_id if class_id is not None else i
            color_rgb = _palette_color(int(idx))
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

            cv2.rectangle(scene, (x1, y1), (x2, y2), color_bgr, self.thickness)

            if skip_label:
                continue

            text = f"{class_id}" if (labels is None or len(labels) != n) else labels[i]

            (text_w, text_h), _ = cv2.getTextSize(
                text=text,
                fontFace=_FONT,
                fontScale=self.text_scale,
                thickness=self.text_thickness,
            )

            if not self.avoid_overlap:
                text_x = x1 + self.text_padding
                text_y = y1 - self.text_padding
                bg = (
                    x1,
                    y1 - 2 * self.text_padding - text_h,
                    x1 + 2 * self.text_padding + text_w,
                    y1,
                )
            else:
                text_x, text_y, *bg_coords = get_optimal_label_pos(
                    self.text_padding,
                    text_w,
                    text_h,
                    x1,
                    y1,
                    x2,
                    y2,
                    detections,
                    image_size,
                )
                bg = tuple(bg_coords)  # type: ignore[assignment]

            cv2.rectangle(
                scene,
                (bg[0], bg[1]),
                (bg[2], bg[3]),
                color_bgr,
                cv2.FILLED,
            )
            luminance = 0.299 * color_rgb[0] + 0.587 * color_rgb[1] + 0.114 * color_rgb[2]
            text_color = (0, 0, 0) if luminance > 160 else (255, 255, 255)
            cv2.putText(
                scene,
                text=text,
                org=(text_x, text_y),
                fontFace=_FONT,
                fontScale=self.text_scale,
                color=text_color,
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )
        return scene


def _candidate_overlaps(
    detections: Any,
    bg: tuple[int, int, int, int],
    image_size: tuple[int, int] | None,
) -> bool:
    """True if a label background overlaps existing detections or runs off-image."""
    bg_box = list(bg)
    for i in range(len(detections)):
        det = detections.xyxy[i].astype(int)
        if iou(bg_box, det.tolist()) > 0.3:
            return True
    return image_size is not None and (
        bg[0] < 0 or bg[2] > image_size[0] or bg[1] < 0 or bg[3] > image_size[1]
    )


def get_optimal_label_pos(
    text_padding: int,
    text_width: int,
    text_height: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,  # noqa: ARG001 - kept for upstream signature compatibility
    detections: Any,
    image_size: tuple[int, int] | None,
) -> tuple[int, int, int, int, int, int]:
    """Try four candidate label positions; return the first non-overlapping one.

    Order: top-left, outer-left, outer-right, top-right. Falls back to
    top-right if all four overlap (matches upstream behaviour).
    """
    # 1. top left
    text_x = x1 + text_padding
    text_y = y1 - text_padding
    bg = (
        x1,
        y1 - 2 * text_padding - text_height,
        x1 + 2 * text_padding + text_width,
        y1,
    )
    if not _candidate_overlaps(detections, bg, image_size):
        return text_x, text_y, *bg

    # 2. outer left
    text_x = x1 - text_padding - text_width
    text_y = y1 + text_padding + text_height
    bg = (
        x1 - 2 * text_padding - text_width,
        y1,
        x1,
        y1 + 2 * text_padding + text_height,
    )
    if not _candidate_overlaps(detections, bg, image_size):
        return text_x, text_y, *bg

    # 3. outer right
    text_x = x2 + text_padding
    text_y = y1 + text_padding + text_height
    bg = (
        x2,
        y1,
        x2 + 2 * text_padding + text_width,
        y1 + 2 * text_padding + text_height,
    )
    if not _candidate_overlaps(detections, bg, image_size):
        return text_x, text_y, *bg

    # 4. top right (fallback)
    text_x = x2 - text_padding - text_width
    text_y = y1 - text_padding
    bg = (
        x2 - 2 * text_padding - text_width,
        y1 - 2 * text_padding - text_height,
        x2,
        y1,
    )
    return text_x, text_y, *bg


def annotate(
    image_source: np.ndarray,
    boxes_cxcywh_normalised: Any,
    phrases: Sequence[str | int],
    *,
    text_scale: float,
    text_padding: int = 5,
    text_thickness: int = 2,
    thickness: int = 3,
) -> tuple[np.ndarray, dict[str, Sequence[float]]]:
    """Annotate ``image_source`` with the supplied normalised ``cxcywh`` boxes.

    Replaces upstream ``annotate``. The ``logits`` argument is dropped
    because upstream never used it (labels were ``[str(i) ...]``).
    """
    try:
        import torch
        from torchvision.ops import box_convert
    except ImportError as exc:  # pragma: no cover
        msg = "annotate() requires the 'caption' or 'yolo' extra (for torch/torchvision)."
        raise ImportError(msg) from exc

    h, w, _ = image_source.shape
    boxes = boxes_cxcywh_normalised * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    xywh = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()

    class _Detections:
        # Minimal duck-typed stand-in for supervision.Detections to avoid a
        # hard dependency on the supervision package in core.
        def __init__(self, xyxy: np.ndarray) -> None:
            self.xyxy = xyxy
            self.class_id: np.ndarray | None = None

        def __len__(self) -> int:
            return len(self.xyxy)

    detections = _Detections(xyxy)
    labels = [str(i) for i in range(len(boxes))]

    annotator = BoxAnnotator(
        text_scale=text_scale,
        text_padding=text_padding,
        text_thickness=text_thickness,
        thickness=thickness,
    )
    annotated = image_source.copy()
    annotated = annotator.annotate(annotated, detections, labels=labels, image_size=(w, h))

    label_coordinates = {f"{phrase}": v for phrase, v in zip(phrases, xywh, strict=False)}
    return annotated, label_coordinates
