"""Tests for omniparser_eval.models.

OpenAI HTTP layer is mocked via respx + httpx. No real OpenAI calls are made.
"""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any, ClassVar

import httpx
import openai
import pytest
import respx
from PIL import Image

from omniparser_eval.models import (
    GroundModel,
    OpenAIGroundModel,
    denormalize_coords,
    extract_first_bounding_box,
    extract_first_point,
    reformat_messages,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(width: int = 1024, height: int = 768) -> Image.Image:
    return Image.new("RGB", (width, height), color=(255, 255, 255))


def _make_chat_completion(content: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-2024-05-13",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            },
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestExtractors:
    def test_extract_bbox(self) -> None:
        assert extract_first_bounding_box("foo [[0.1,0.2,0.3,0.4]] bar") == [0.1, 0.2, 0.3, 0.4]

    def test_extract_bbox_int(self) -> None:
        assert extract_first_bounding_box("[[1,2,3,4]]") == [1.0, 2.0, 3.0, 4.0]

    def test_extract_bbox_none(self) -> None:
        assert extract_first_bounding_box("nothing here") is None

    def test_extract_point(self) -> None:
        assert extract_first_point("[[0.5,0.6]]") == [0.5, 0.6]

    def test_extract_point_none(self) -> None:
        assert extract_first_point("no coords") is None


class TestDenormalize:
    def test_point(self) -> None:
        assert denormalize_coords([0.5, 0.5], (1000, 800)) == [500, 400]

    def test_bbox(self) -> None:
        assert denormalize_coords([0.0, 0.0, 1.0, 1.0], (1000, 800)) == [0, 0, 1000, 800]

    def test_rounds_to_nearest(self) -> None:
        # 0.1234 * 1000 = 123.4 → 123; 0.5678 * 800 = 454.24 → 454
        assert denormalize_coords([0.1234, 0.5678], (1000, 800)) == [123, 454]

    def test_invalid_length(self) -> None:
        with pytest.raises(ValueError, match="2- or 4-tuple"):
            denormalize_coords([0.1, 0.2, 0.3], (1000, 800))


class TestReformatMessages:
    def test_text_and_icon(self) -> None:
        elements = [
            {"type": "text", "content": "Hello", "bbox": [0, 0, 10, 10]},
            {"type": "icon", "content": "Save button", "bbox": [10, 10, 20, 20]},
        ]
        out = reformat_messages(elements)  # type: ignore[arg-type]
        assert '<p id=0 class="text" alt="Hello">' in out
        assert '<img id=1 class="icon" alt="Save button">' in out

    def test_mutates_idx(self) -> None:
        elements: list[dict[str, Any]] = [
            {"type": "text", "content": "a", "bbox": [0, 0, 1, 1]},
            {"type": "text", "content": "b", "bbox": [1, 1, 2, 2]},
        ]
        reformat_messages(elements)  # type: ignore[arg-type]
        assert elements[0]["idx"] == 0
        assert elements[1]["idx"] == 1


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_openai_ground_model_satisfies_protocol() -> None:
    client = openai.OpenAI(api_key="test", base_url="http://test/v1")
    model = OpenAIGroundModel(client=client, parser=None)
    assert isinstance(model, GroundModel)


def test_init_initializes_override_generation_config() -> None:
    """Upstream bug fix: override_generation_config is used in upstream lines
    174/259/343 but never initialized. We initialize it to {"temperature": 0.0}.
    """
    client = openai.OpenAI(api_key="test", base_url="http://test/v1")
    model = OpenAIGroundModel(client=client, parser=None)
    assert model.override_generation_config == {"temperature": 0.0}
    model.set_generation_config(temperature=0.7, top_p=0.9)
    assert model.override_generation_config["temperature"] == 0.7
    assert model.override_generation_config["top_p"] == 0.9


# ---------------------------------------------------------------------------
# ground_allow_negative
# ---------------------------------------------------------------------------


@respx.mock
def test_ground_allow_negative_returns_pixel_bbox() -> None:
    base = "http://test/v1"
    respx.post(f"{base}/chat/completions").mock(
        return_value=httpx.Response(200, json=_make_chat_completion("[[0.1,0.2,0.3,0.4]]")),
    )
    client = openai.OpenAI(api_key="test", base_url=base)
    model = OpenAIGroundModel(client=client, parser=None)
    image = _make_image(1000, 800)

    result = model.ground_allow_negative("click save", image)

    assert result["result"] == "positive"
    assert result["bbox"] == [100, 160, 300, 320]  # absolute pixels
    # point computed as bbox center
    assert result["point"] == [200, 240]


@respx.mock
def test_ground_allow_negative_handles_negative() -> None:
    base = "http://test/v1"
    respx.post(f"{base}/chat/completions").mock(
        return_value=httpx.Response(200, json=_make_chat_completion("Target not existent")),
    )
    client = openai.OpenAI(api_key="test", base_url=base)
    model = OpenAIGroundModel(client=client, parser=None)

    result = model.ground_allow_negative("non-existent thing", _make_image())

    assert result["result"] == "negative"
    assert result["bbox"] is None
    assert result["point"] is None
    assert "not existent" in str(result["raw_response"]).lower()


@respx.mock
def test_ground_allow_negative_handles_no_match() -> None:
    """If the model response has no bbox AND no 'not existent' marker."""
    base = "http://test/v1"
    respx.post(f"{base}/chat/completions").mock(
        return_value=httpx.Response(200, json=_make_chat_completion("Sorry, can't help.")),
    )
    client = openai.OpenAI(api_key="test", base_url=base)
    model = OpenAIGroundModel(client=client, parser=None)

    result = model.ground_allow_negative("instruction", _make_image())
    assert result["result"] == "negative"
    assert result["bbox"] is None


@respx.mock
def test_ground_allow_negative_uses_temperature_override() -> None:
    base = "http://test/v1"
    captured: dict[str, Any] = {}

    def _record(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_make_chat_completion("Target not existent"))

    respx.post(f"{base}/chat/completions").mock(side_effect=_record)
    client = openai.OpenAI(api_key="test", base_url=base)
    model = OpenAIGroundModel(client=client, parser=None)
    model.set_generation_config(temperature=0.7)

    model.ground_allow_negative("x", _make_image())
    assert captured["body"]["temperature"] == 0.7


# ---------------------------------------------------------------------------
# ground_with_uncertainty
# ---------------------------------------------------------------------------


@respx.mock
def test_ground_with_uncertainty_handles_not_found() -> None:
    """Upstream uses 'not found' here despite prompt saying 'not existent';
    we accept either to be robust."""
    base = "http://test/v1"
    respx.post(f"{base}/chat/completions").mock(
        return_value=httpx.Response(200, json=_make_chat_completion("Target not found")),
    )
    client = openai.OpenAI(api_key="test", base_url=base)
    model = OpenAIGroundModel(client=client, parser=None)
    result = model.ground_with_uncertainty("x", _make_image())
    assert result["result"] == "negative"


@respx.mock
def test_ground_with_uncertainty_returns_pixel_bbox() -> None:
    base = "http://test/v1"
    respx.post(f"{base}/chat/completions").mock(
        return_value=httpx.Response(200, json=_make_chat_completion("[[0.0,0.0,0.5,0.5]]")),
    )
    client = openai.OpenAI(api_key="test", base_url=base)
    model = OpenAIGroundModel(client=client, parser=None)
    result = model.ground_with_uncertainty("x", _make_image(2000, 1000))
    assert result["bbox"] == [0, 0, 1000, 500]
    assert result["point"] == [500, 250]


# ---------------------------------------------------------------------------
# ground_only_positive (parser is required)
# ---------------------------------------------------------------------------


def test_ground_only_positive_requires_parser() -> None:
    client = openai.OpenAI(api_key="test", base_url="http://test/v1")
    model = OpenAIGroundModel(client=client, parser=None)
    with pytest.raises(ValueError, match="requires a parser"):
        model.ground_only_positive("x", _make_image())


@respx.mock
def test_ground_only_positive_with_mock_parser() -> None:
    """Verify the full ground_only_positive flow with both OpenAI and parser mocked.

    The parser stub returns one element and one label_coordinates entry;
    OpenAI returns a structured dict naming that bbox id; we assert the
    final pixel-coordinate output.
    """
    base = "http://test/v1"
    response_dict = {
        "Analysis": "click the save button",
        "Click BBox ID": "0",
    }
    respx.post(f"{base}/chat/completions").mock(
        return_value=httpx.Response(200, json=_make_chat_completion(json.dumps(response_dict))),
    )
    client = openai.OpenAI(api_key="test", base_url=base)

    # Build a SoM image as base64 PNG (small).
    som_buf = BytesIO()
    _make_image(100, 100).save(som_buf, format="PNG")
    import base64 as _b64

    som_b64 = _b64.b64encode(som_buf.getvalue()).decode()

    class _StubParseResult:
        annotated_image_b64 = som_b64
        # label_coordinates uses normalized (x, y, w, h) per upstream convention
        label_coordinates: ClassVar[dict[str, list[float]]] = {"0": [0.1, 0.2, 0.1, 0.2]}
        elements: ClassVar[list[dict[str, Any]]] = [
            {"type": "icon", "content": "Save", "bbox": [0.1, 0.2, 0.2, 0.4]},
        ]

    class _StubParser:
        def parse(self, image: Any) -> _StubParseResult:
            return _StubParseResult()

    model = OpenAIGroundModel(client=client, parser=_StubParser())  # type: ignore[arg-type]
    image = _make_image(1000, 800)

    result = model.ground_only_positive("save the file", image)
    assert result["result"] == "positive"
    # label_coords [0.1, 0.2, 0.1, 0.2] on 1000x800 → x=100, y=160, w=100, h=160
    # bbox xyxy = [100, 160, 200, 320]; point = [150, 240]
    assert result["bbox"] == [100, 160, 200, 320]
    assert result["point"] == [150, 240]
    assert result["raw_response"] == response_dict


@respx.mock
def test_ground_only_positive_falls_back_to_regex_dict_extractor() -> None:
    """If GPT-4o wraps the dict in prose, we fall back to brace-balanced extraction."""
    base = "http://test/v1"
    prose = (
        'Sure, here is my answer: {"Analysis": "click here", "Click BBox ID": "0"} Hope this helps!'
    )
    respx.post(f"{base}/chat/completions").mock(
        return_value=httpx.Response(200, json=_make_chat_completion(prose)),
    )
    client = openai.OpenAI(api_key="test", base_url=base)

    class _StubParseResult:
        annotated_image_b64 = ""
        label_coordinates: ClassVar[dict[str, list[float]]] = {"0": [0.0, 0.0, 0.5, 0.5]}
        elements: ClassVar[list[Any]] = []

    class _StubParser:
        def parse(self, image: Any) -> _StubParseResult:
            return _StubParseResult()

    model = OpenAIGroundModel(client=client, parser=_StubParser())  # type: ignore[arg-type]
    result = model.ground_only_positive("x", _make_image(1000, 800))
    assert result["result"] == "positive"
    # 0,0,0.5,0.5 on 1000x800 → x=0,y=0,w=500,h=400 → xyxy = [0,0,500,400], point=[250,200]
    assert result["bbox"] == [0, 0, 500, 400]
    assert result["point"] == [250, 200]
