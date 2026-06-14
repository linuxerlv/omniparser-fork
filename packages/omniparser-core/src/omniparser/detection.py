"""Detector backends for omniparser-core.

The pipeline only depends on the :class:`Detector` Protocol; the
Ultralytics YOLO implementation is guarded by the ``[yolo]`` extra
because Ultralytics is **AGPL-3.0** (see WEIGHTS_LICENSE.md). Users who
do not install the extra can still import the rest of the package and
plug in their own detector.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

__all__ = ["DetectionResult", "Detector", "UltralyticsYoloDetector"]

_logger = logging.getLogger(__name__)

# (boxes_xyxy_pixels, confidences) where boxes is shape (N, 4).
DetectionResult = tuple[np.ndarray, np.ndarray]


@runtime_checkable
class Detector(Protocol):
    """Pluggable icon detector.

    Implementations take an image and return per-detection bounding boxes
    in pixel-space ``(x1, y1, x2, y2)`` together with confidence scores.
    """

    def predict(
        self,
        image: PILImage,
        *,
        box_threshold: float,
        iou_threshold: float = 0.7,
        imgsz: tuple[int, int] | None = None,
    ) -> DetectionResult:
        """Run detection on ``image``."""
        ...


class UltralyticsYoloDetector:
    """Detector backend powered by `Ultralytics <https://docs.ultralytics.com>`_.

    .. warning::

       Ultralytics is licensed under AGPL-3.0. Embedding it in a network
       service triggers AGPL §13 obligations on the *entire combined
       work*. See ``WEIGHTS_LICENSE.md`` for compliance options.
    """

    def __init__(self, model_path: str | Path) -> None:
        """Load YOLO weights from ``model_path``; supports ``.pt`` files."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - import-time guard
            msg = (
                "UltralyticsYoloDetector requires the 'yolo' extra. "
                "Install with: pip install 'omniparser-core[yolo]'\n"
                "Note: Ultralytics is AGPL-3.0; see WEIGHTS_LICENSE.md."
            )
            raise ImportError(msg) from exc

        self._model = YOLO(str(model_path))
        _logger.debug("Loaded YOLO weights from %s", model_path)

    def predict(
        self,
        image: PILImage,
        *,
        box_threshold: float,
        iou_threshold: float = 0.7,
        imgsz: tuple[int, int] | None = None,
    ) -> DetectionResult:
        """Run YOLO inference on ``image`` and return ``(boxes_xyxy, conf)``."""
        kwargs: dict[str, Any] = {
            "source": image,
            "conf": box_threshold,
            "iou": iou_threshold,
            "verbose": False,
        }
        if imgsz is not None:
            kwargs["imgsz"] = imgsz

        result = self._model.predict(**kwargs)[0]
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        return boxes_xyxy, conf
