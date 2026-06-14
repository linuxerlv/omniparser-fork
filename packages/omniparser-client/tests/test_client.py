"""Tests for :class:`OmniparserClient` / :class:`AsyncOmniparserClient`.

These tests use :class:`httpx.MockTransport` so they do not need a running
server. Wire shape is covered both ways: requests are inspected before they
hit the transport, responses are forged and asserted on the parsed model.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from omniparser_client import (
    AsyncOmniparserClient,
    OmniparserClient,
    OmniparserHTTPError,
    OmniparserSchemaError,
    ParseRequest,
    encode_image,
)


@pytest.fixture(scope="session")
def png_bytes() -> bytes:
    img = Image.new("RGB", (4, 4), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def png_path(tmp_path_factory: pytest.TempPathFactory, png_bytes: bytes) -> Path:
    path = tmp_path_factory.mktemp("client") / "screen.png"
    path.write_bytes(png_bytes)
    return path


_PARSE_OK_BODY = {
    "schema_version": "1",
    "annotated_image_base64": "ZmFrZQ==",
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
    "latency_ms": 12.34,
}


_HEALTH_OK_BODY = {
    "status": "ok",
    "version": "0.1.0.dev0",
    "schema_version": "1",
    "backends": {"detector": "yolo", "captioner": "florence2", "ocr": "easyocr", "device": "cpu"},
}


def _parse_handler_factory(captured: dict[str, object]) -> object:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        # GET /healthz has no body; only POST /parse carries JSON.
        raw = request.content
        captured["body"] = json.loads(raw.decode()) if raw else None
        if request.url.path == "/healthz":
            return httpx.Response(200, json=_HEALTH_OK_BODY)
        return httpx.Response(200, json=_PARSE_OK_BODY)

    return handler


# ---------------------------------------------------------------------------
# encode_image helper
# ---------------------------------------------------------------------------


def test_encode_image_from_bytes(png_bytes: bytes) -> None:
    encoded = encode_image(png_bytes)
    assert base64.b64decode(encoded) == png_bytes


def test_encode_image_from_path(png_path: Path, png_bytes: bytes) -> None:
    encoded = encode_image(png_path)
    assert base64.b64decode(encoded) == png_bytes


def test_encode_image_from_pil(png_bytes: bytes) -> None:
    image = Image.open(io.BytesIO(png_bytes))
    encoded = encode_image(image)
    # Round-trip: decoded base64 must be a valid PNG.
    Image.open(io.BytesIO(base64.b64decode(encoded))).verify()


def test_encode_image_string_path_round_trip(png_path: Path, png_bytes: bytes) -> None:
    encoded = encode_image(str(png_path))
    assert base64.b64decode(encoded) == png_bytes


def test_encode_image_string_passthrough() -> None:
    # Looks like base64 already; not a path → returned verbatim.
    payload = "ZmFrZQ=="
    assert encode_image(payload) == payload


def test_encode_image_rejects_unknown_type() -> None:
    with pytest.raises(TypeError, match="unsupported image type"):
        encode_image(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Sync OmniparserClient
# ---------------------------------------------------------------------------


def test_sync_parse_round_trip(png_bytes: bytes) -> None:
    captured: dict[str, object] = {}
    transport = httpx.MockTransport(_parse_handler_factory(captured))  # type: ignore[arg-type]

    with OmniparserClient("http://test", transport=transport) as client:
        result = client.parse(png_bytes, box_threshold=0.5, imgsz=1024)

    assert captured["method"] == "POST"
    assert captured["url"] == "http://test/parse"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["box_threshold"] == 0.5
    assert body["imgsz"] == 1024
    assert body["iou_threshold"] is None
    # Round-trip the encoded image.
    assert base64.b64decode(body["image_base64"]) == png_bytes

    assert result.schema_version == "1"
    assert len(result.elements) == 2
    assert result.elements[1].content is None


def test_sync_health(png_bytes: bytes) -> None:
    captured: dict[str, object] = {}
    transport = httpx.MockTransport(_parse_handler_factory(captured))  # type: ignore[arg-type]

    with OmniparserClient("http://test", transport=transport) as client:
        health = client.health()

    assert health.status == "ok"
    assert health.backends["detector"] == "yolo"


def test_sync_http_error_envelope() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": "HTTPException",
                "detail": "bad image",
                "schema_version": "1",
            },
        )

    transport = httpx.MockTransport(handler)
    with (
        OmniparserClient("http://test", transport=transport) as client,
        pytest.raises(
            OmniparserHTTPError,
        ) as exc_info,
    ):
        client.parse(b"\x00\x00")

    assert exc_info.value.status_code == 400
    assert exc_info.value.body is not None
    assert exc_info.value.body.detail == "bad image"


def test_sync_schema_mismatch_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        # Missing required fields → ValidationError → OmniparserSchemaError.
        return httpx.Response(200, json={"unexpected": True})

    transport = httpx.MockTransport(handler)
    with (
        OmniparserClient("http://test", transport=transport) as client,
        pytest.raises(
            OmniparserSchemaError,
        ),
    ):
        client.parse(b"\x00")


# ---------------------------------------------------------------------------
# Async AsyncOmniparserClient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_parse_round_trip(png_bytes: bytes) -> None:
    captured: dict[str, object] = {}
    transport = httpx.MockTransport(_parse_handler_factory(captured))  # type: ignore[arg-type]

    async with AsyncOmniparserClient("http://test", transport=transport) as client:
        result = await client.parse(png_bytes, iou_threshold=0.2)

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["iou_threshold"] == 0.2
    assert result.elements[0].type == "text"


@pytest.mark.asyncio
async def test_async_health() -> None:
    captured: dict[str, object] = {}
    transport = httpx.MockTransport(_parse_handler_factory(captured))  # type: ignore[arg-type]

    async with AsyncOmniparserClient("http://test", transport=transport) as client:
        health = await client.health()

    assert health.status == "ok"


# ---------------------------------------------------------------------------
# Pure schema sanity
# ---------------------------------------------------------------------------


def test_parse_request_extra_field_rejected() -> None:
    with pytest.raises(ValueError, match="extra"):
        ParseRequest.model_validate({"image_base64": "abc", "made_up": True})


def test_parse_request_imgsz_bounds() -> None:
    with pytest.raises(ValueError, match=r"greater_than|less_than|imgsz"):
        ParseRequest(image_base64="abc", imgsz=2)
