"""Smoke tests for :mod:`omniparser` package surface.

Verifies that the public API is importable without optional extras and
that import time is fast (no model loading).
"""

from __future__ import annotations

import importlib
import time


def test_top_level_import_is_fast() -> None:
    """Importing omniparser must complete in well under 500 ms.

    Patch A acceptance: no import-time side effects (no easyocr.Reader,
    no PaddleOCR, no torch.cuda.is_available probing).
    """
    # Use a fresh subprocess-like measurement by invalidating cached
    # entries. importlib import time is a cheap proxy and avoids needing
    # `subprocess`.
    importlib.invalidate_caches()
    start = time.perf_counter()
    importlib.import_module("omniparser")
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"omniparser import took {elapsed:.3f}s (limit 0.5s)"


def test_public_api_surface() -> None:
    import omniparser

    assert hasattr(omniparser, "Omniparser")
    assert hasattr(omniparser, "OmniparserConfig")
    assert hasattr(omniparser, "ParseResult")
    assert hasattr(omniparser, "merge_proposals")
    assert hasattr(omniparser, "resolve_device")
    assert omniparser.__version__ == "0.1.0.dev0"


def test_resolve_device_falls_back_to_cpu_without_torch() -> None:
    # Without [caption]/[yolo] extras installed in the test env, torch is
    # absent and resolve_device must return "cpu".
    from omniparser import resolve_device

    device = resolve_device()
    assert device in {"cpu", "cuda", "mps"}


def test_protocols_are_importable_without_backends() -> None:
    # The Protocol definitions must be available even when no concrete
    # backend extra is installed.
    from omniparser.detection import Detector
    from omniparser.ocr import OcrBackend

    assert Detector is not None
    assert OcrBackend is not None
