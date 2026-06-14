"""Synchronous and asynchronous HTTP clients for ``omniparser-server``.

The clients are thin wrappers around :class:`httpx.Client` /
:class:`httpx.AsyncClient` that:

* Build :class:`omniparser_client.schemas.ParseRequest` from caller arguments.
* POST to ``/parse`` and validate the response against
  :class:`omniparser_client.schemas.ParseResponse`.
* Translate non-2xx responses into :class:`OmniparserHTTPError`.
* Expose ``health()`` for liveness probing.

Image inputs are accepted as raw bytes, base64 strings, file paths, or
``PIL.Image.Image`` objects (when Pillow is available); the helper
:func:`encode_image` does the conversion and is exported for callers that
want to control the encoding step explicitly.
"""

from __future__ import annotations

import base64
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self

import httpx
from pydantic import ValidationError

from omniparser_client.errors import OmniparserHTTPError, OmniparserSchemaError
from omniparser_client.schemas import (
    ErrorResponse,
    HealthResponse,
    ParseRequest,
    ParseResponse,
)

ImageInput = "bytes | str | Path | PILImage"

DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0)


class _SupportsSave(Protocol):
    """Minimal duck-typed Pillow image surface (avoids hard dep on Pillow)."""

    def save(self, fp: Any, format: str = ...) -> None: ...


def encode_image(image: bytes | str | Path | _SupportsSave) -> str:
    """Encode ``image`` as base64 (no ``data:`` prefix).

    Accepts:

    * ``bytes`` — already-encoded image bytes (PNG/JPEG).
    * ``str`` — either a filesystem path or an already-base64-encoded string.
      A heuristic differentiates them: if it points to an existing file, it is
      treated as a path; otherwise it is returned as-is (the server tolerates
      ``data:image/...,`` prefixes).
    * ``pathlib.Path`` — read from disk.
    * Anything with a ``save()`` method — assumed to be a Pillow image; saved
      as PNG to a buffer.
    """
    if isinstance(image, (bytes, bytearray)):
        return base64.b64encode(image).decode("ascii")

    if isinstance(image, Path):
        return base64.b64encode(image.read_bytes()).decode("ascii")

    if isinstance(image, str):
        candidate = Path(image)
        if candidate.exists():
            return base64.b64encode(candidate.read_bytes()).decode("ascii")
        return image

    if hasattr(image, "save"):
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    msg = f"unsupported image type: {type(image).__name__}"
    raise TypeError(msg)


def _decode_response(resp: httpx.Response) -> ParseResponse:
    if resp.is_success:
        try:
            return ParseResponse.model_validate_json(resp.content)
        except ValidationError as exc:
            msg = f"server returned a body that does not match ParseResponse: {exc}"
            raise OmniparserSchemaError(msg) from exc

    body: ErrorResponse | None
    try:
        body = ErrorResponse.model_validate_json(resp.content)
    except ValidationError:
        body = None
    raise OmniparserHTTPError(resp.status_code, body, resp.text)


def _decode_health(resp: httpx.Response) -> HealthResponse:
    if not resp.is_success:
        body: ErrorResponse | None
        try:
            body = ErrorResponse.model_validate_json(resp.content)
        except ValidationError:
            body = None
        raise OmniparserHTTPError(resp.status_code, body, resp.text)
    try:
        return HealthResponse.model_validate_json(resp.content)
    except ValidationError as exc:
        raise OmniparserSchemaError(f"invalid health body: {exc}") from exc


def _build_request(
    image: bytes | str | Path | _SupportsSave,
    *,
    box_threshold: float | None,
    iou_threshold: float | None,
    use_paddleocr: bool | None,
    imgsz: int | None,
) -> ParseRequest:
    return ParseRequest(
        image_base64=encode_image(image),
        box_threshold=box_threshold,
        iou_threshold=iou_threshold,
        use_paddleocr=use_paddleocr,
        imgsz=imgsz,
    )


class OmniparserClient(AbstractContextManager["OmniparserClient"]):
    """Synchronous HTTP client for ``omniparser-server``.

    Designed to be used as a context manager so the underlying
    :class:`httpx.Client` is closed deterministically.

    Example:
        >>> with OmniparserClient("http://localhost:8000") as client:
        ...     result = client.parse(Path("screenshot.png"))
        ...     for el in result.elements:
        ...         print(el.type, el.bbox, el.content)
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: httpx.Timeout | float | None = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
            headers=headers,
        )

    @property
    def http(self) -> httpx.Client:
        """Expose the underlying httpx client for advanced use cases."""
        return self._client

    def parse(
        self,
        image: bytes | str | Path | _SupportsSave,
        *,
        box_threshold: float | None = None,
        iou_threshold: float | None = None,
        use_paddleocr: bool | None = None,
        imgsz: int | None = None,
    ) -> ParseResponse:
        """Send a ``POST /parse`` and return the validated response."""
        body = _build_request(
            image,
            box_threshold=box_threshold,
            iou_threshold=iou_threshold,
            use_paddleocr=use_paddleocr,
            imgsz=imgsz,
        )
        resp = self._client.post(
            "/parse",
            content=body.model_dump_json(),
            headers={"content-type": "application/json"},
        )
        return _decode_response(resp)

    def health(self) -> HealthResponse:
        """Send a ``GET /healthz`` and return the validated response."""
        return _decode_health(self._client.get("/healthz"))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class AsyncOmniparserClient(AbstractAsyncContextManager["AsyncOmniparserClient"]):
    """Async sibling of :class:`OmniparserClient`."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: httpx.Timeout | float | None = DEFAULT_TIMEOUT,
        transport: httpx.AsyncBaseTransport | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
            headers=headers,
        )

    @property
    def http(self) -> httpx.AsyncClient:
        return self._client

    async def parse(
        self,
        image: bytes | str | Path | _SupportsSave,
        *,
        box_threshold: float | None = None,
        iou_threshold: float | None = None,
        use_paddleocr: bool | None = None,
        imgsz: int | None = None,
    ) -> ParseResponse:
        body = _build_request(
            image,
            box_threshold=box_threshold,
            iou_threshold=iou_threshold,
            use_paddleocr=use_paddleocr,
            imgsz=imgsz,
        )
        resp = await self._client.post(
            "/parse",
            content=body.model_dump_json(),
            headers={"content-type": "application/json"},
        )
        return _decode_response(resp)

    async def health(self) -> HealthResponse:
        resp = await self._client.get("/healthz")
        return _decode_health(resp)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
