"""HTTP wire schemas for the OmniParser inference service.

These models define the contract between ``omniparser-server`` and
``omniparser-client``. They are intentionally minimal and decoupled from the
in-process :class:`omniparser.ParseResult` so the wire format can evolve
independently of the Python API.

Versioning policy
-----------------

* Wire schema is versioned via the ``Server-Schema-Version`` response header
  and the ``schema_version`` field on :class:`ParseResponse`.
* Bumping the schema is a breaking change while we are pre-1.0; clients are
  expected to pin a compatible server version.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION: Literal["1"] = "1"


class ParsedElementModel(BaseModel):
    """One element produced by the parser.

    The ``bbox`` is normalized to ``[0, 1]`` (xyxy) so callers can scale it to
    any rendered size without knowing the original screenshot dimensions.
    ``content`` is nullable to mirror the in-process :class:`omniparser.ParsedElement`
    contract: when captioning is disabled or fails, the element is still emitted
    with ``content=None`` so the bounding box is not lost.
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
    """Inbound request body for ``POST /parse``."""

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
    """Outbound response body for ``POST /parse``."""

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
