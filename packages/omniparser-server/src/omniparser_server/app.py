"""FastAPI application factory for ``omniparser-server``.

The application is constructed via :func:`create_app` so tests can build a
fresh instance with custom settings and dependency overrides without going
through the CLI.

Endpoints
---------

* ``GET /healthz`` — liveness/readiness probe; returns 200 once the parser is
  loaded (or immediately when ``eager_load=False`` and the server has booted).
* ``GET /probe`` — alias for ``/healthz``, kept for compatibility with the
  upstream client that called ``/probe/``.
* ``POST /parse`` — main inference endpoint.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import io
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from omniparser_server.dependencies import ParserState, get_parser_state
from omniparser_server.logging_config import configure_logging, get_logger
from omniparser_server.schemas import (
    SCHEMA_VERSION,
    ErrorResponse,
    HealthResponse,
    ParsedElementModel,
    ParseRequest,
    ParseResponse,
)
from omniparser_server.settings import ServerSettings

logger = get_logger(__name__)

_DATA_URL_PREFIX = "data:image/"


def _server_version() -> str:
    try:
        return version("omniparser-server")
    except PackageNotFoundError:
        return "0.0.0+unknown"


def _decode_image(image_base64: str) -> Image.Image:
    payload = image_base64
    if payload.startswith(_DATA_URL_PREFIX):
        _, _, payload = payload.partition(",")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"image_base64 is not valid base64: {exc}",
        ) from exc
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"image_base64 does not decode to a valid image: {exc}",
        ) from exc
    return image.convert("RGB") if image.mode != "RGB" else image


def _config_with_overrides(
    base: object,
    *,
    box_threshold: float | None,
    iou_threshold: float | None,
    use_paddleocr: bool | None,
    imgsz: int | None,
) -> object:
    """Return a copy of ``base`` :class:`OmniparserConfig` with per-request overrides applied.

    ``base`` is typed as ``object`` to keep this module import-light; the runtime
    type is :class:`omniparser.OmniparserConfig` (a frozen dataclass).
    """
    overrides: dict[str, object] = {}
    if box_threshold is not None:
        overrides["box_threshold"] = box_threshold
    if iou_threshold is not None:
        overrides["iou_threshold"] = iou_threshold
    if use_paddleocr is not None:
        overrides["use_paddleocr"] = use_paddleocr
    if imgsz is not None:
        overrides["imgsz"] = (imgsz, imgsz)
    if not overrides:
        return base
    return dataclasses.replace(base, **overrides)  # type: ignore[type-var]


def create_app(
    settings: ServerSettings | None = None,
    *,
    configure_logging_on_startup: bool = True,
) -> FastAPI:
    """Build a fresh FastAPI app.

    Tests should pass an explicit ``settings`` and ``configure_logging_on_startup=False``
    so they do not clobber the test runner's logging.
    """
    cfg = settings if settings is not None else ServerSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if configure_logging_on_startup:
            configure_logging(level=cfg.log_level, fmt=cfg.log_format)
        state = ParserState(cfg)
        app.state.parser_state = state
        logger.info(
            "server_starting",
            host=cfg.host,
            port=cfg.port,
            eager_load=cfg.eager_load,
            schema_version=SCHEMA_VERSION,
        )
        await state.startup()
        try:
            yield
        finally:
            logger.info("server_stopping")
            await state.shutdown()

    app = FastAPI(
        title="OmniParser Inference Server",
        version=_server_version(),
        lifespan=lifespan,
        responses={
            400: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )

    if cfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cfg.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        body = ErrorResponse(error=exc.__class__.__name__, detail=str(exc.detail))
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    async def healthz(
        state: Annotated[ParserState, Depends(get_parser_state)],
    ) -> HealthResponse:
        status_value = "ok" if state.is_ready() or not state.settings.eager_load else "starting"
        return HealthResponse(
            status=status_value,  # type: ignore[arg-type]
            version=_server_version(),
            backends=state.backends,
        )

    app.add_api_route("/healthz", endpoint=healthz, response_model=HealthResponse, methods=["GET"])
    app.add_api_route("/probe", endpoint=healthz, response_model=HealthResponse, methods=["GET"])

    @app.post("/parse", response_model=ParseResponse)
    async def parse(
        body: ParseRequest,
        state: Annotated[ParserState, Depends(get_parser_state)],
    ) -> ParseResponse:
        image = _decode_image(body.image_base64)
        parser = await state.ensure_loaded()

        from omniparser.pipeline import parse_screen

        cfg_overridden = _config_with_overrides(
            parser.config,
            box_threshold=body.box_threshold,
            iou_threshold=body.iou_threshold,
            use_paddleocr=body.use_paddleocr,
            imgsz=body.imgsz,
        )

        started = time.perf_counter()
        try:
            result = parse_screen(
                image,
                detector=parser.detector,
                ocr_backend=parser.ocr_backend,
                captioner_bundle=parser.captioner,
                config=cfg_overridden,  # type: ignore[arg-type]
            )
        except Exception as exc:
            logger.exception("parse_failed", error=str(exc))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"parse_failed: {type(exc).__name__}: {exc}",
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        elements: list[ParsedElementModel] = []
        for el in result.elements:
            bbox = el["bbox"]
            elements.append(
                ParsedElementModel(
                    type=el["type"],  # type: ignore[arg-type]
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    interactivity=bool(el["interactivity"]),
                    content=el["content"],
                    source=el["source"],  # type: ignore[arg-type]
                ),
            )

        logger.info(
            "parse_ok",
            latency_ms=round(latency_ms, 2),
            element_count=len(elements),
        )

        return ParseResponse(
            annotated_image_base64=result.annotated_image_b64,
            elements=elements,
            latency_ms=latency_ms,
        )

    # FastAPI uses ``parse`` via the decorator above; assign to ``_`` so static
    # analyzers do not flag the closure as unused.
    _ = parse
    return app
