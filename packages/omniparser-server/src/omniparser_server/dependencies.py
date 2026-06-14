"""Lifespan-managed singletons for the FastAPI app.

The :class:`omniparser.Omniparser` instance is expensive to construct (loads
detector and captioner weights, may also touch CUDA), so it lives for the
lifetime of the process. We instantiate it once during the FastAPI ``lifespan``
hook and expose it through a tiny accessor that endpoints depend on.

The accessor pattern (rather than a module-level global) keeps tests honest:
``app.dependency_overrides`` can swap a fake parser without monkey-patching
imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Request, status

if TYPE_CHECKING:
    from omniparser import Omniparser
    from omniparser_server.settings import ServerSettings


class ParserState:
    """Owns the long-lived :class:`Omniparser` and exposes a backend probe.

    Stored on ``app.state.parser_state`` so dependency callables can fetch it
    via the request scope without importing module-level globals.
    """

    __slots__ = ("_backends", "_parser", "_settings")

    def __init__(self, settings: ServerSettings) -> None:
        self._settings = settings
        self._parser: Omniparser | None = None
        self._backends: dict[str, str] = {
            "detector": settings.detector_backend,
            "captioner": settings.captioner_backend,
            "ocr": settings.ocr_backend,
            "device": settings.device,
        }

    @property
    def settings(self) -> ServerSettings:
        return self._settings

    @property
    def backends(self) -> dict[str, str]:
        return dict(self._backends)

    def is_ready(self) -> bool:
        return self._parser is not None

    def get(self) -> Omniparser:
        if self._parser is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="parser is not initialized yet",
            )
        return self._parser

    def install(self, parser: Omniparser) -> None:
        """Inject a pre-built parser. Used by tests via dependency overrides."""
        self._parser = parser

    async def startup(self) -> None:
        """Build the parser if eager loading is enabled.

        When ``eager_load=False`` (the default), the first ``/parse`` request
        is responsible for triggering construction via :meth:`ensure_loaded`.
        """
        if self._settings.eager_load:
            await self.ensure_loaded()

    async def ensure_loaded(self) -> Omniparser:
        if self._parser is not None:
            return self._parser
        self._parser = build_parser(self._settings)
        return self._parser

    async def shutdown(self) -> None:
        self._parser = None


def build_parser(settings: ServerSettings) -> Omniparser:
    """Construct an :class:`Omniparser` from server settings.

    Imports are deferred so importing :mod:`omniparser_server.dependencies`
    does not pull torch/transformers into processes that only want to read
    settings (e.g. ``omniparser-server --print-settings``).
    """
    from omniparser import Omniparser, OmniparserConfig, resolve_device

    device = resolve_device(settings.device)

    detector = _build_detector(settings)
    captioner_bundle = _build_captioner_bundle(settings, device=device)
    ocr_backend = _build_ocr_backend(settings)

    config = OmniparserConfig(
        box_threshold=settings.box_threshold_default,
        iou_threshold=settings.iou_threshold_default,
        text_threshold=settings.ocr_text_threshold,
        use_paddleocr=(settings.ocr_backend == "paddleocr"),
        output_coord_in_ratio=settings.output_coord_in_ratio,
        imgsz=_imgsz_tuple(settings.detector_imgsz),
        batch_size=settings.captioner_batch_size,
    )

    return Omniparser(
        detector=detector,
        ocr_backend=ocr_backend,
        captioner=captioner_bundle,
        config=config,
    )


def _imgsz_tuple(value: int) -> tuple[int, int]:
    return (value, value)


def _build_detector(settings: ServerSettings) -> Any:
    if settings.detector_weights is None:
        msg = (
            "detector_weights is not configured; set OMNIPARSER_DETECTOR_WEIGHTS "
            "to the path of a YOLO .pt file"
        )
        raise RuntimeError(msg)

    from omniparser.detection import UltralyticsYoloDetector

    return UltralyticsYoloDetector(settings.detector_weights)


def _build_captioner_bundle(settings: ServerSettings, *, device: str) -> dict[str, Any] | None:
    if settings.captioner_weights is None:
        msg = (
            "captioner_weights is not configured; set OMNIPARSER_CAPTIONER_WEIGHTS "
            "to the directory containing a Florence-2 / BLIP-2 / Phi-3V checkpoint"
        )
        raise RuntimeError(msg)

    from omniparser.captioning import load_caption_model

    return load_caption_model(
        model_name=settings.captioner_backend,
        model_name_or_path=str(settings.captioner_weights),
        device=device,
    )


def _build_ocr_backend(settings: ServerSettings) -> Any | None:
    if settings.ocr_backend == "none":
        return None
    if settings.ocr_backend == "easyocr":
        from omniparser.ocr import EasyOcrBackend

        return EasyOcrBackend()
    if settings.ocr_backend == "paddleocr":
        from omniparser.ocr import PaddleOcrBackend

        return PaddleOcrBackend()
    # Defensive: pydantic Literal validation should make this unreachable, but
    # we keep an explicit error path for runtime safety against monkey-patched
    # settings objects in tests.
    msg = f"unsupported ocr_backend: {settings.ocr_backend!r}"  # type: ignore[unreachable]
    raise RuntimeError(msg)


def get_parser_state(request: Request) -> ParserState:
    """FastAPI dependency that returns the per-app :class:`ParserState`."""
    state: ParserState = request.app.state.parser_state
    return state
