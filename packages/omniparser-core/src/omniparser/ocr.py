"""OCR backends for omniparser-core.

The pipeline only depends on the :class:`OcrBackend` Protocol; concrete
implementations are guarded by optional extras:

* ``omniparser-core[easyocr]`` → :class:`EasyOcrBackend`
* ``omniparser-core[paddleocr]`` → :class:`PaddleOcrBackend`

This eliminates upstream's import-time side effect (patch A): the
``easyocr.Reader([...])`` and ``PaddleOCR(...)`` constructors used to fire
during ``import util.utils``, blocking startup for ~3 s and downloading
~500 MB of models on first run. Backends here are constructed lazily by
the caller and may be cached behind ``functools.lru_cache`` if desired.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "EasyOcrBackend",
    "OcrBackend",
    "OcrResult",
    "PaddleOcrBackend",
    "get_xywh",
    "get_xyxy",
]

_logger = logging.getLogger(__name__)

# (text, bbox) where bbox is (x1, y1, x2, y2) in pixel coordinates.
OcrResult = tuple[list[str], list[tuple[int, int, int, int]]]


@runtime_checkable
class OcrBackend(Protocol):
    """Pluggable OCR backend.

    Implementations must accept either a path or a PIL image and return a
    parallel list of recognised strings and their bounding boxes. Boxes
    are returned in pixel-space ``(x1, y1, x2, y2)``.
    """

    def read(
        self,
        image: str | Path | Image.Image,
        *,
        text_threshold: float = 0.5,
    ) -> OcrResult:
        """Run OCR on ``image`` and return ``(texts, bboxes)``."""
        ...


def get_xywh(quad: Sequence[Sequence[float]]) -> tuple[int, int, int, int]:
    """Convert an OCR engine's 4-corner quad to ``(x, y, w, h)``.

    The input is the canonical OCR shape ``[[x0,y0],[x1,y1],[x2,y2],[x3,y3]]``
    where corners 0 and 2 are diagonally opposite.
    """
    x = int(quad[0][0])
    y = int(quad[0][1])
    w = int(quad[2][0] - quad[0][0])
    h = int(quad[2][1] - quad[0][1])
    return x, y, w, h


def get_xyxy(quad: Sequence[Sequence[float]]) -> tuple[int, int, int, int]:
    """Convert an OCR engine's 4-corner quad to ``(x1, y1, x2, y2)``."""
    return int(quad[0][0]), int(quad[0][1]), int(quad[2][0]), int(quad[2][1])


def _coerce_to_rgb_array(image: str | Path | Image.Image) -> tuple[np.ndarray, tuple[int, int]]:
    """Load image (or coerce a PIL image) to an RGB ``np.ndarray``.

    Returns ``(array, (width, height))``. RGBA inputs are flattened to RGB
    because EasyOCR rejects 4-channel inputs.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if image.mode == "RGBA" or image.mode != "RGB":
        image = image.convert("RGB")
    return np.asarray(image), image.size


class EasyOcrBackend:
    """OCR backend powered by `EasyOCR <https://github.com/JaidedAI/EasyOCR>`_.

    Construction loads ~70 MB of detection + recognition weights on first
    use. Reuse a single instance across calls to amortise the cost.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        *,
        gpu: bool = False,
        **reader_kwargs: Any,
    ) -> None:
        """Construct an EasyOCR reader; loads ~70MB of weights on first call."""
        try:
            import easyocr
        except ImportError as exc:  # pragma: no cover - import-time guard
            msg = (
                "EasyOcrBackend requires the 'easyocr' extra. Install with: "
                "pip install 'omniparser-core[easyocr]'"
            )
            raise ImportError(msg) from exc

        self._languages = languages or ["en"]
        self._reader = easyocr.Reader(self._languages, gpu=gpu, **reader_kwargs)

    def read(
        self,
        image: str | Path | Image.Image,
        *,
        text_threshold: float = 0.5,  # noqa: ARG002 - threshold is applied by EasyOCR internally
        **easyocr_kwargs: Any,
    ) -> OcrResult:
        """Run EasyOCR on ``image`` and return ``(texts, bboxes_xyxy)``."""
        np_image, _ = _coerce_to_rgb_array(image)
        results = self._reader.readtext(np_image, **easyocr_kwargs)
        texts: list[str] = []
        bboxes: list[tuple[int, int, int, int]] = []
        for quad, text, _conf in results:
            texts.append(text)
            bboxes.append(get_xyxy(quad))
        return texts, bboxes


class PaddleOcrBackend:
    """OCR backend powered by `PaddleOCR <https://github.com/PaddlePaddle/PaddleOCR>`_.

    Defaults mirror upstream OmniParser's tuned configuration: angle
    classification disabled, slow-but-accurate detection mode, dilation
    enabled. ``use_gpu=False`` because Paddle's CUDA build conflicts with
    PyTorch's in the same process.
    """

    def __init__(
        self,
        *,
        lang: str = "en",
        use_gpu: bool = False,
        **paddle_kwargs: Any,
    ) -> None:
        """Construct a PaddleOCR reader with upstream's tuned defaults."""
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - import-time guard
            msg = (
                "PaddleOcrBackend requires the 'paddleocr' extra. Install with: "
                "pip install 'omniparser-core[paddleocr]'"
            )
            raise ImportError(msg) from exc

        defaults: dict[str, Any] = {
            "lang": lang,
            "use_angle_cls": False,
            "use_gpu": use_gpu,
            "show_log": False,
            "max_batch_size": 1024,
            "use_dilation": True,
            "det_db_score_mode": "slow",
            "rec_batch_num": 1024,
        }
        defaults.update(paddle_kwargs)
        self._reader = PaddleOCR(**defaults)

    def read(
        self,
        image: str | Path | Image.Image,
        *,
        text_threshold: float = 0.5,
    ) -> OcrResult:
        """Run PaddleOCR on ``image``; filters by ``text_threshold`` confidence."""
        np_image, _ = _coerce_to_rgb_array(image)
        result = self._reader.ocr(np_image, cls=False)
        if not result or result[0] is None:
            return [], []
        rows = result[0]
        texts: list[str] = []
        bboxes: list[tuple[int, int, int, int]] = []
        for quad, (text, conf) in rows:
            if conf <= text_threshold:
                continue
            texts.append(text)
            bboxes.append(get_xyxy(quad))
        return texts, bboxes
