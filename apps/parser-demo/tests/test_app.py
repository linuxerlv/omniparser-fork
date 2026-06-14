"""Tests for ``parser_demo.app``.

These tests exercise the pure-Python helpers and verify that the Gradio
factory builds without loading any models. They do NOT call ``demo.launch``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from parser_demo.app import DemoConfig, _format_elements, build_demo

# Gradio creates a ProactorEventLoop on Windows and never tears it down
# cleanly, which surfaces as PytestUnraisableExceptionWarning under our
# strict filterwarnings policy. The leak is in Gradio, not in our code, so
# we silence it here only — production code never sees this path because
# the demo process exits long before GC runs the loop's finalizer.
pytestmark = pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnraisableExceptionWarning",
    "ignore::ResourceWarning",
)


def test_demo_config_frozen() -> None:
    cfg = DemoConfig(
        detector_weights=Path("dummy.pt"),
        captioner_weights=Path("dummy"),
    )
    with pytest.raises(AttributeError):
        cfg.share = True  # type: ignore[misc]


def test_demo_config_defaults() -> None:
    cfg = DemoConfig(
        detector_weights=Path("dummy.pt"),
        captioner_weights=Path("dummy"),
    )
    assert cfg.captioner_backend == "florence2"
    assert cfg.server_name == "127.0.0.1"
    assert cfg.server_port == 7861
    assert cfg.share is False  # safety default — must not flip silently


def test_format_elements_empty() -> None:
    assert _format_elements([]) == "<no elements detected>"


def test_format_elements_renders_text_and_icon() -> None:
    elements = [
        {
            "type": "text",
            "bbox": [0.1, 0.2, 0.15, 0.23],
            "interactivity": False,
            "content": "Hello",
            "source": "box_ocr_content_ocr",
        },
        {
            "type": "icon",
            "bbox": [0.5, 0.5, 0.6, 0.6],
            "interactivity": True,
            "content": None,
            "source": "box_yolo_content_yolo",
        },
    ]
    rendered = _format_elements(elements)
    assert rendered == "0. [Text] Hello\n1. [Icon] <no caption>"


def test_build_demo_does_not_load_models() -> None:
    """The factory must not pull any ML weights at construction time."""

    cfg = DemoConfig(
        detector_weights=Path("nonexistent.pt"),
        captioner_weights=Path("nonexistent"),
    )
    demo = build_demo(cfg)
    # If this returns at all, no weights were loaded — paths are bogus.
    assert demo is not None
