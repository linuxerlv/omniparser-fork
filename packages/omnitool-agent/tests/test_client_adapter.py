"""Tests for the ``omnitool-agent`` adapter.

We mock at the httpx transport layer via ``respx`` so the tests neither
need a running server nor depend on the real omniparser-client internals.
The wire schema is exercised via :class:`omniparser_client.ParseResponse`,
keeping these tests honest if the schema ever changes.
"""

from __future__ import annotations

import base64
import io

import httpx
import pytest
import respx
from PIL import Image

from omnitool_agent import (
    ImageDecodeError,
    OmnitoolParserClient,
    ParsedScreen,
)


@pytest.fixture(scope="module")
def png_b64() -> str:
    """Return a valid 16x9 PNG screenshot encoded as base64."""

    img = Image.new("RGB", (16, 9), color=(31, 41, 59))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture
def parse_response_payload() -> dict[str, object]:
    """Wire-level JSON the server would return for a ``POST /parse``."""

    return {
        "schema_version": "1",
        "annotated_image_base64": "ZmFrZS1hbm5vdGF0ZWQ=",
        "elements": [
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
                "content": None,
                "source": "box_yolo_content_yolo",
            },
        ],
        "latency_ms": 123.5,
    }


def test_parse_screenshot_returns_legacy_shape(
    png_b64: str,
    parse_response_payload: dict[str, object],
) -> None:
    with respx.mock(base_url="http://test") as router:
        router.post("/parse").respond(200, json=parse_response_payload)
        with OmnitoolParserClient("http://test") as client:
            result: ParsedScreen = client.parse_screenshot(png_b64)

    # Legacy keys all present.
    assert set(result.keys()) == {
        "som_image_base64",
        "parsed_content_list",
        "latency",
        "width",
        "height",
        "original_screenshot_base64",
        "screenshot_uuid",
        "screen_info",
    }


def test_parse_screenshot_preserves_wire_fields(
    png_b64: str,
    parse_response_payload: dict[str, object],
) -> None:
    with respx.mock(base_url="http://test") as router:
        router.post("/parse").respond(200, json=parse_response_payload)
        with OmnitoolParserClient("http://test") as client:
            result = client.parse_screenshot(png_b64)

    assert result["som_image_base64"] == "ZmFrZS1hbm5vdGF0ZWQ="
    assert result["latency"] == pytest.approx(0.1235)  # ms -> s
    assert result["original_screenshot_base64"] == png_b64
    assert result["width"] == 16
    assert result["height"] == 9


def test_parsed_content_list_has_idx_and_legacy_fields(
    png_b64: str,
    parse_response_payload: dict[str, object],
) -> None:
    with respx.mock(base_url="http://test") as router:
        router.post("/parse").respond(200, json=parse_response_payload)
        with OmnitoolParserClient("http://test") as client:
            result = client.parse_screenshot(png_b64)

    elements = result["parsed_content_list"]
    assert [el["idx"] for el in elements] == [0, 1]
    assert elements[0]["type"] == "text"
    assert elements[0]["bbox"] == [0.10, 0.20, 0.15, 0.23]
    assert elements[0]["interactivity"] is False
    assert elements[0]["content"] == "Hello"
    assert elements[1]["type"] == "icon"
    assert elements[1]["content"] is None  # captioning disabled is preserved


def test_screen_info_renders_text_and_icon_lines(
    png_b64: str,
    parse_response_payload: dict[str, object],
) -> None:
    with respx.mock(base_url="http://test") as router:
        router.post("/parse").respond(200, json=parse_response_payload)
        with OmnitoolParserClient("http://test") as client:
            result = client.parse_screenshot(png_b64)

    info = result["screen_info"]
    assert "ID: 0, Text: Hello" in info
    assert "ID: 1, Icon: " in info  # None content renders as empty string
    assert info.endswith("\n")


def test_screenshot_uuid_is_unique_per_call(
    png_b64: str,
    parse_response_payload: dict[str, object],
) -> None:
    with respx.mock(base_url="http://test") as router:
        router.post("/parse").respond(200, json=parse_response_payload)
        with OmnitoolParserClient("http://test") as client:
            a = client.parse_screenshot(png_b64)
            b = client.parse_screenshot(png_b64)

    assert a["screenshot_uuid"] != b["screenshot_uuid"]


def test_accepts_bytes_input(
    parse_response_payload: dict[str, object],
) -> None:
    img = Image.new("RGB", (8, 4), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw = buf.getvalue()

    with respx.mock(base_url="http://test") as router:
        router.post("/parse").respond(200, json=parse_response_payload)
        with OmnitoolParserClient("http://test") as client:
            result = client.parse_screenshot(raw)

    assert result["width"] == 8
    assert result["height"] == 4
    # original_screenshot_base64 should now be a base64 string of the bytes.
    assert base64.b64decode(result["original_screenshot_base64"]) == raw


def test_accepts_data_url_prefix(
    png_b64: str,
    parse_response_payload: dict[str, object],
) -> None:
    data_url = f"data:image/png;base64,{png_b64}"

    with respx.mock(base_url="http://test") as router:
        router.post("/parse").respond(200, json=parse_response_payload)
        with OmnitoolParserClient("http://test") as client:
            result = client.parse_screenshot(data_url)

    assert result["width"] == 16
    assert result["height"] == 9


def test_invalid_base64_raises_image_decode_error() -> None:
    # No need to hit the network: size-decode happens after the wire call,
    # so we still mock it but the failure should be raised from our adapter.
    payload = {
        "schema_version": "1",
        "annotated_image_base64": "ZmFrZQ==",
        "elements": [],
        "latency_ms": 1.0,
    }
    with respx.mock(base_url="http://test") as router:
        router.post("/parse").respond(200, json=payload)
        with (
            OmnitoolParserClient("http://test") as client,
            pytest.raises(ImageDecodeError),
        ):
            client.parse_screenshot("!!!not-base64!!!")


def test_http_error_propagates_from_omniparser_client(
    png_b64: str,
) -> None:
    from omniparser_client import OmniparserHTTPError

    error_payload = {
        "schema_version": "1",
        "error": "validation_error",
        "detail": "image_base64 is required",
    }
    with respx.mock(base_url="http://test") as router:
        router.post("/parse").respond(422, json=error_payload)
        with (
            OmnitoolParserClient("http://test") as client,
            pytest.raises(OmniparserHTTPError),
        ):
            client.parse_screenshot(png_b64)


def test_close_releases_underlying_http_client(
    png_b64: str,
    parse_response_payload: dict[str, object],
) -> None:
    with respx.mock(base_url="http://test") as router:
        router.post("/parse").respond(200, json=parse_response_payload)
        client = OmnitoolParserClient("http://test")
        client.parse_screenshot(png_b64)
        client.close()
    # A second close must be a no-op (idempotent), matching httpx semantics.
    client.close()


def test_custom_transport_is_used(
    png_b64: str,
    parse_response_payload: dict[str, object],
) -> None:
    """Smoke test that we forward the ``transport`` kwarg through."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/parse"
        return httpx.Response(200, json=parse_response_payload)

    transport = httpx.MockTransport(handler)
    with OmnitoolParserClient("http://test", transport=transport) as client:
        result = client.parse_screenshot(png_b64)
    assert result["width"] == 16
