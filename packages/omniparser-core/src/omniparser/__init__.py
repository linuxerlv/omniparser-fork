"""omniparser — pure inference library for screen-element parsing.

This package provides:

* :class:`Omniparser` — the high-level facade
* :class:`OmniparserConfig` — typed configuration
* :class:`ParseResult` — the typed return value
* Pluggable Protocols: :class:`Detector`, :class:`OcrBackend`

Concrete backends live in submodules and are guarded by extras:

* ``omniparser.detection.UltralyticsYoloDetector`` — needs ``[yolo]``
* ``omniparser.ocr.EasyOcrBackend`` — needs ``[easyocr]``
* ``omniparser.ocr.PaddleOcrBackend`` — needs ``[paddleocr]``
* ``omniparser.captioning.load_caption_model`` — needs ``[caption]``

Importing this top-level package has no model-loading side effects; it
should complete in well under 500 ms.
"""

from __future__ import annotations

from omniparser.api import Omniparser, resolve_device
from omniparser.pipeline import (
    OmniparserConfig,
    ParsedElement,
    ParseResult,
    merge_proposals,
    parse_screen,
)

__version__ = "0.1.0.dev0"

__all__ = [
    "Omniparser",
    "OmniparserConfig",
    "ParseResult",
    "ParsedElement",
    "__version__",
    "merge_proposals",
    "parse_screen",
    "resolve_device",
]
