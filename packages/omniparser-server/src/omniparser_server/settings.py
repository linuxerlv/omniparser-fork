"""Runtime configuration for ``omniparser-server``.

Configuration is sourced (in increasing precedence) from:

1. Built-in defaults declared on :class:`ServerSettings`.
2. ``OMNIPARSER_*`` environment variables.
3. A ``.env`` file in the working directory (if present).
4. CLI flags wired by :mod:`omniparser_server.cli`.

All defaults are safe for development. Production deployments must override at
minimum :attr:`ServerSettings.detector_weights` and
:attr:`ServerSettings.captioner_weights` with paths that exist on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DeviceLiteral = Literal["auto", "cuda", "mps", "cpu"]
CaptionBackendLiteral = Literal["florence2", "blip2", "phi3v"]
OcrBackendLiteral = Literal["easyocr", "paddleocr", "none"]
LogFormatLiteral = Literal["json", "console"]


class ServerSettings(BaseSettings):
    """Process-wide settings.

    Only fields that genuinely affect server behavior live here. Inference
    knobs that the *caller* should be able to tune per-request (such as
    ``box_threshold``) are exposed on :class:`omniparser_server.schemas.ParseRequest`
    instead, with these values acting as defaults.
    """

    model_config = SettingsConfigDict(
        env_prefix="OMNIPARSER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        frozen=True,
    )

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: LogFormatLiteral = "json"

    device: DeviceLiteral = "auto"

    detector_backend: Literal["yolo"] = "yolo"
    detector_weights: Path | None = None
    detector_imgsz: int = Field(default=640, ge=64, le=4096)

    captioner_backend: CaptionBackendLiteral = "florence2"
    captioner_weights: Path | None = None
    captioner_batch_size: int = Field(default=128, ge=1, le=2048)

    ocr_backend: OcrBackendLiteral = "easyocr"
    ocr_paragraph: bool = False
    ocr_text_threshold: float = Field(default=0.9, ge=0.0, le=1.0)

    box_threshold_default: float = Field(default=0.05, ge=0.0, le=1.0)
    iou_threshold_default: float = Field(default=0.10, ge=0.0, le=1.0)
    output_coord_in_ratio: bool = True

    request_timeout_s: float = Field(default=120.0, gt=0.0)
    cors_origins: tuple[str, ...] = ()

    eager_load: bool = False

    @field_validator("detector_weights", "captioner_weights")
    @classmethod
    def _expand_path(cls, value: Path | None) -> Path | None:
        return value.expanduser().resolve() if value is not None else None
