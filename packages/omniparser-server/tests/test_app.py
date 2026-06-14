"""End-to-end tests for the ``/parse``, ``/healthz``, ``/probe`` endpoints."""

from __future__ import annotations

from typing import Any

import httpx
import pytest


@pytest.mark.asyncio
async def test_healthz_ok(app_with_fake: Any) -> None:
    transport = httpx.ASGITransport(app=app_with_fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["schema_version"] == "1"
    assert set(body["backends"]) == {"detector", "captioner", "ocr", "device"}


@pytest.mark.asyncio
async def test_probe_alias(app_with_fake: Any) -> None:
    transport = httpx.ASGITransport(app=app_with_fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/probe")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_parse_returns_schema(app_with_fake: Any, png_b64: str) -> None:
    transport = httpx.ASGITransport(app=app_with_fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/parse", json={"image_base64": png_b64})

    assert resp.status_code == 200
    body = resp.json()
    assert body["schema_version"] == "1"
    assert body["annotated_image_base64"] == "ZmFrZQ=="
    assert body["latency_ms"] >= 0
    assert len(body["elements"]) == 2
    assert body["elements"][0] == {
        "type": "text",
        "bbox": [0.10, 0.20, 0.15, 0.23],
        "interactivity": False,
        "content": "Hello",
        "source": "box_ocr_content_ocr",
    }
    assert body["elements"][1]["content"] is None  # null caption survives wire trip


@pytest.mark.asyncio
async def test_parse_accepts_data_url_prefix(app_with_fake: Any, png_b64: str) -> None:
    transport = httpx.ASGITransport(app=app_with_fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/parse",
            json={"image_base64": f"data:image/png;base64,{png_b64}"},
        )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_parse_rejects_garbage_base64(app_with_fake: Any) -> None:
    transport = httpx.ASGITransport(app=app_with_fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/parse", json={"image_base64": "!!!not-base64!!!"})
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] == "HTTPException"
    assert "base64" in body["detail"].lower()


@pytest.mark.asyncio
async def test_parse_rejects_non_image_payload(app_with_fake: Any) -> None:
    transport = httpx.ASGITransport(app=app_with_fake)
    # Valid base64 but the bytes are not an image.
    import base64

    payload = base64.b64encode(b"hello world").decode("ascii")
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/parse", json={"image_base64": payload})
    assert resp.status_code == 400
    assert "image" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_parse_overrides_propagate(
    app_with_fake: Any,
    captured_calls: dict[str, Any],
    png_b64: str,
) -> None:
    transport = httpx.ASGITransport(app=app_with_fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/parse",
            json={
                "image_base64": png_b64,
                "box_threshold": 0.42,
                "iou_threshold": 0.13,
                "imgsz": 1280,
                "use_paddleocr": True,
            },
        )
    assert resp.status_code == 200
    cfg = captured_calls["kwargs"]["config"]
    assert cfg.box_threshold == pytest.approx(0.42)
    assert cfg.iou_threshold == pytest.approx(0.13)
    assert cfg.imgsz == (1280, 1280)
    assert cfg.use_paddleocr is True


@pytest.mark.asyncio
async def test_parse_without_overrides_keeps_base_config(
    app_with_fake: Any,
    captured_calls: dict[str, Any],
    png_b64: str,
) -> None:
    transport = httpx.ASGITransport(app=app_with_fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/parse", json={"image_base64": png_b64})
    assert resp.status_code == 200
    # Same OmniparserConfig instance flowed through unchanged.
    cfg = captured_calls["kwargs"]["config"]
    from omniparser import OmniparserConfig

    assert cfg == OmniparserConfig()


@pytest.mark.asyncio
async def test_parse_extra_field_rejected(app_with_fake: Any, png_b64: str) -> None:
    transport = httpx.ASGITransport(app=app_with_fake)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/parse",
            json={"image_base64": png_b64, "made_up_field": True},
        )
    assert resp.status_code == 422  # pydantic validation, before our handler runs
