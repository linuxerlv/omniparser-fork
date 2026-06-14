"""Pytest fixtures for ``omniparser-server`` tests."""

from __future__ import annotations

import base64
import io
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image

from omniparser_server.app import create_app
from omniparser_server.dependencies import ParserState
from omniparser_server.settings import ServerSettings


@dataclass(slots=True)
class FakeParseResult:
    """Mirror of :class:`omniparser.ParseResult` used as a test double."""

    annotated_image_b64: str
    elements: list[dict[str, Any]]
    label_coordinates: dict[str, list[float]]


@pytest.fixture(scope="session")
def png_b64() -> str:
    """Return a valid 4x4 PNG screenshot encoded as base64."""

    img = Image.new("RGB", (4, 4), color=(127, 127, 127))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture
def fake_parse_result() -> FakeParseResult:
    return FakeParseResult(
        annotated_image_b64="ZmFrZQ==",  # base64("fake")
        elements=[
            {
                "type": "text",
                "bbox": [0.10, 0.20, 0.15, 0.23],
                "interactivity": False,
                "content": "Hello",
                "source": "box_ocr_content_ocr",
            },
            {
                "type": "icon",
                "bbox": [0.50, 0.50, 0.60, 0.60],
                "interactivity": True,
                "content": None,  # captioning disabled
                "source": "box_yolo_content_yolo",
            },
        ],
        label_coordinates={"0": [0.1, 0.2, 0.15, 0.23]},
    )


@pytest.fixture
def fake_parser(fake_parse_result: FakeParseResult) -> MagicMock:
    """Build a duck-typed Omniparser stand-in.

    Has the attributes :func:`parse_screen` reads on the real one
    (``detector``, ``ocr_backend``, ``captioner``, ``config``) plus a ``parse``
    method that the server does NOT call directly — present only so the type
    checker is happy when reading ``parser.config``.
    """

    from omniparser import OmniparserConfig

    parser = MagicMock(name="Omniparser")
    parser.detector = MagicMock(name="Detector")
    parser.ocr_backend = MagicMock(name="OcrBackend")
    parser.captioner = None
    parser.config = OmniparserConfig()
    return parser


@pytest.fixture
def settings() -> ServerSettings:
    return ServerSettings(
        host="127.0.0.1",
        port=8000,
        log_level="WARNING",
        log_format="console",
        eager_load=False,
        detector_weights=None,
        captioner_weights=None,
    )


@pytest.fixture
def app_with_fake(
    settings: ServerSettings,
    fake_parser: MagicMock,
    fake_parse_result: FakeParseResult,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Any]:
    """Build a FastAPI app whose parser is a pre-installed fake.

    We monkeypatch ``omniparser.pipeline.parse_screen`` so the ``/parse``
    endpoint returns deterministic data without touching any model. The fake
    :class:`ParserState` is wired onto ``app.state`` directly because
    ``httpx.ASGITransport`` does NOT trigger the FastAPI lifespan, and we want
    the test to be independent of lifespan plumbing anyway.
    """

    import omniparser.pipeline as pipeline_module

    captured: dict[str, Any] = {}

    def fake_parse_screen(image: Any, **kwargs: Any) -> FakeParseResult:
        captured["image"] = image
        captured["kwargs"] = kwargs
        return fake_parse_result

    monkeypatch.setattr(pipeline_module, "parse_screen", fake_parse_screen)

    app = create_app(settings, configure_logging_on_startup=False)
    state = ParserState(settings)
    state.install(fake_parser)
    app.state.parser_state = state
    app.state.test_capture = captured
    yield app


@pytest.fixture
def captured_calls(app_with_fake: Any) -> dict[str, Any]:
    return app_with_fake.state.test_capture  # type: ignore[no-any-return]
