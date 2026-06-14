"""Wire schemas for the OmniParser HTTP client.

These models MUST stay byte-compatible with
:mod:`omniparser_server.schemas`. The two packages keep separate copies so
that ``omniparser-client`` does not pull FastAPI / uvicorn / pydantic-settings
into environments that only want to talk to a remote parser.

A contract test in ``packages/omniparser-server/tests/test_schema_contract.py``
asserts the JSON Schema of every model here matches the server's, so any
divergence is caught in CI.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION: Literal["1"] = "1"


class ParsedElementModel(BaseModel):
    """One element produced by the parser.

    The ``bbox`` is normalized to ``[0, 1]`` (xyxy) so callers can scale it to
    any rendered size without knowing the original screenshot dimensions.
    ``content`` is nullable: when captioning is disabled or fails, the element
    is still emitted with ``content=None`` so the bounding box is not lost.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["text", "icon"]
    bbox: Annotated[
        tuple[float, float, float, float],
        Field(
            description="Normalized xyxy bounding box in [0, 1].",
            examples=[(0.10, 0.20, 0.15, 0.23)],
        ),
    ]
    interactivity: bool
    content: str | None = None
    source: Literal[
        "box_ocr_content_ocr",
        "box_yolo_content_yolo",
        "box_yolo_content_ocr",
    ]


class ParseRequest(BaseModel):
    """Outbound request body for ``POST /parse``."""

    model_config = ConfigDict(extra="forbid")

    image_base64: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Screenshot encoded as base64. May include the standard "
                "``data:image/png;base64,`` prefix; both forms are accepted."
            ),
        ),
    ]
    box_threshold: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    iou_threshold: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    use_paddleocr: bool | None = None
    imgsz: Annotated[int | None, Field(ge=64, le=4096)] = None


class ParseResponse(BaseModel):
    """Inbound response body for ``POST /parse``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = SCHEMA_VERSION
    annotated_image_base64: str
    elements: list[ParsedElementModel]
    latency_ms: float


class HealthResponse(BaseModel):
    """Liveness/readiness response shared by ``/healthz`` and ``/probe``."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "starting", "degraded"]
    version: str
    schema_version: Literal["1"] = SCHEMA_VERSION
    backends: dict[str, str]


class ErrorResponse(BaseModel):
    """Uniform error envelope returned for all 4xx/5xx errors."""

    model_config = ConfigDict(extra="forbid")

    error: str
    detail: str | None = None
    schema_version: Literal["1"] = SCHEMA_VERSION
