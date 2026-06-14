"""High-level :class:`Omniparser` facade.

Replaces ``upstream/util/omniparser.py``. Differences:

* Constructor accepts a typed :class:`OmniparserConfig` and explicit
  backend instances rather than a ``Dict[str, Any]`` config blob.
* ``parse`` accepts ``PIL.Image | numpy.ndarray | str | Path`` directly
  — base64 decoding lives in the server layer (where it belongs) rather
  than in core.
* No ``print()`` calls (patch C); device probing supports CUDA → MPS →
  CPU (patch B); class declared as ``class Omniparser:`` rather than
  ``class Omniparser(object):`` (patch G).
* The ``BOX_TRESHOLD`` typo is fixed to ``box_threshold`` (patch F).

This is the breaking-change interface: there is no shim for upstream's
13-arg ``get_som_labeled_img`` because we are pre-1.0 and the rewrite is
the whole point of the package.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from omniparser.pipeline import OmniparserConfig, ParseResult, parse_screen

if TYPE_CHECKING:
    import numpy as np

    from omniparser.detection import Detector
    from omniparser.ocr import OcrBackend

__all__ = ["Omniparser", "resolve_device"]

_logger = logging.getLogger(__name__)


def resolve_device(preferred: str | None = None) -> str:
    """Return the best available torch device name.

    Falls back ``cuda → mps → cpu`` (patch B). Importing torch is
    deferred so that callers without ``[caption]`` or ``[yolo]`` extras
    can still import this module.
    """
    if preferred and preferred != "auto":
        return preferred
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Omniparser:
    """High-level facade combining detector + captioner + OCR.

    Construction is cheap (just stashes references); model loading is the
    caller's responsibility, so tests can substitute mocks freely.

    Example:
        >>> from omniparser import Omniparser, OmniparserConfig
        >>> from omniparser.detection import UltralyticsYoloDetector
        >>> from omniparser.ocr import EasyOcrBackend
        >>> from omniparser.captioning import load_caption_model
        >>> parser = Omniparser(
        ...     detector=UltralyticsYoloDetector("weights/icon_detect/model.pt"),
        ...     ocr_backend=EasyOcrBackend(),
        ...     captioner=load_caption_model("florence2", "weights/icon_caption_florence"),
        ...     config=OmniparserConfig(box_threshold=0.05),
        ... )
        >>> result = parser.parse("screenshot.png")
        >>> print(result.elements[:3])
    """

    def __init__(
        self,
        *,
        detector: Detector,
        ocr_backend: OcrBackend | None = None,
        captioner: dict[str, Any] | None = None,
        config: OmniparserConfig | None = None,
    ) -> None:
        """Stash references to the injected backends; no model loading happens here."""
        self.detector = detector
        self.ocr_backend = ocr_backend
        self.captioner = captioner
        self.config = config or OmniparserConfig()
        _logger.debug(
            "Omniparser ready (detector=%s, ocr=%s, captioner=%s)",
            type(detector).__name__,
            type(ocr_backend).__name__ if ocr_backend else None,
            "present" if captioner else None,
        )

    def parse(self, image: Image.Image | np.ndarray | str | Path) -> ParseResult:
        """Parse a single screenshot.

        Args:
            image: PIL image, numpy array (HxWxC RGB), filesystem path,
                or pathlib.Path. Base64 strings are not accepted here —
                use :meth:`parse_base64` or decode in the caller.
        """
        pil = _coerce_to_pil(image)

        # Auto-tune draw config from image scale (preserved from upstream).
        if self.config.draw_bbox_config is None:
            ratio = max(pil.size) / 3200
            inferred = {
                "text_scale": 0.8 * ratio,
                "text_thickness": max(int(2 * ratio), 1),
                "text_padding": max(int(3 * ratio), 1),
                "thickness": max(int(3 * ratio), 1),
            }
            cfg = OmniparserConfig(
                box_threshold=self.config.box_threshold,
                iou_threshold=self.config.iou_threshold,
                text_threshold=self.config.text_threshold,
                use_paddleocr=self.config.use_paddleocr,
                use_local_semantics=self.config.use_local_semantics,
                output_coord_in_ratio=self.config.output_coord_in_ratio,
                scale_img=self.config.scale_img,
                imgsz=self.config.imgsz,
                batch_size=self.config.batch_size,
                text_scale=self.config.text_scale,
                text_padding=self.config.text_padding,
                prompt=self.config.prompt,
                draw_bbox_config=inferred,
            )
        else:
            cfg = self.config

        return parse_screen(
            pil,
            detector=self.detector,
            ocr_backend=self.ocr_backend,
            captioner_bundle=self.captioner,
            config=cfg,
        )

    def parse_base64(self, image_b64: str) -> ParseResult:
        """Convenience: decode a base64-encoded PNG/JPEG and parse it.

        Server layers should generally call :meth:`parse` directly with a
        PIL image to avoid a redundant decode step; this method exists
        purely as a back-compat affordance with upstream's wire format.
        """
        return self.parse(Image.open(io.BytesIO(base64.b64decode(image_b64))))


def _coerce_to_pil(image: Image.Image | np.ndarray | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (str, Path)):
        return Image.open(image)
    # numpy array fallback (avoid importing numpy at module top to keep
    # this hot path light when callers only ever pass PIL).
    import numpy as _np_local

    if isinstance(image, _np_local.ndarray):
        return Image.fromarray(image)
    msg = f"Unsupported image type: {type(image).__name__}"  # type: ignore[unreachable]
    raise TypeError(msg)
