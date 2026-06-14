"""Adapter that bridges :mod:`omniparser_client` to the legacy omnitool shape.

The original upstream ``OmniParserClient`` (in
``upstream/omnitool/gradio/agent/llm_utils/omniparserclient.py``) used raw
``requests`` against ``POST /parse/`` and post-processed the JSON into the
shape consumed by the rest of the agent loop. We preserve that consumer
contract while routing all traffic through our typed
:class:`omniparser_client.OmniparserClient`, which adds:

* request/response validation via pydantic
* schema-version pinning (``schema_version="1"``)
* normalized error envelope handling
* sync + async parity (we expose only sync here; async can be added when
  the upstream loop becomes async)

Upstream did NOT compute ``width`` / ``height`` from the request body — it
always called ``get_screenshot()`` first and stuffed the PIL image's size in.
We don't have that helper at adapter level (it lives in the UI app), so we
decode the input PNG/JPEG to recover ``(width, height)``. Pillow is an
optional extra of ``omniparser-client``; we declare it as a hard dependency
of the adapter because the legacy shape is non-negotiable.
"""

from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Self

from omniparser_client import OmniparserClient, ParseResponse
from omnitool_agent.errors import ImageDecodeError
from omnitool_agent.types import LegacyParsedElement, ParsedScreen

if TYPE_CHECKING:
    import httpx


def _decode_size(image_b64: str) -> tuple[int, int]:
    """Recover ``(width, height)`` from a base64-encoded image."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - install-time error
        msg = (
            "Pillow is required to decode screenshot dimensions; install "
            "`omnitool-agent` (which depends on it transitively via "
            "`omniparser-client[pillow]`) or add Pillow explicitly."
        )
        raise ImageDecodeError(msg) from exc

    payload = image_b64.split(",", 1)[-1] if image_b64.startswith("data:") else image_b64
    try:
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, TypeError) as exc:
        msg = "image_base64 is not valid base64"
        raise ImageDecodeError(msg) from exc
    try:
        with Image.open(io.BytesIO(raw)) as im:
            return im.size  # (width, height)
    except Exception as exc:
        msg = f"failed to decode image: {exc}"
        raise ImageDecodeError(msg) from exc


def _format_screen_info(elements: list[LegacyParsedElement]) -> str:
    """Build the human-readable ``screen_info`` string the agent prompt uses.

    Mirrors upstream ``reformat_messages`` (``ID: <i>, Text|Icon: <content>``).
    """
    lines: list[str] = []
    for el in elements:
        idx = el.get("idx")
        kind = el.get("type")
        content = el.get("content") or ""
        if kind == "text":
            lines.append(f"ID: {idx}, Text: {content}")
        elif kind == "icon":
            lines.append(f"ID: {idx}, Icon: {content}")
    return "\n".join(lines) + ("\n" if lines else "")


def _adapt_response(resp: ParseResponse, image_b64: str) -> ParsedScreen:
    """Convert the typed response into the legacy dict shape."""
    width, height = _decode_size(image_b64)
    legacy: list[LegacyParsedElement] = [
        {
            "type": el.type,
            "bbox": list(el.bbox),
            "interactivity": el.interactivity,
            "content": el.content,
            "source": el.source,
            "idx": idx,
        }
        for idx, el in enumerate(resp.elements)
    ]
    return ParsedScreen(
        som_image_base64=resp.annotated_image_base64,
        parsed_content_list=legacy,
        latency=resp.latency_ms / 1000.0,
        width=width,
        height=height,
        original_screenshot_base64=image_b64,
        screenshot_uuid=uuid.uuid4().hex,
        screen_info=_format_screen_info(legacy),
    )


class OmnitoolParserClient:
    """Drop-in replacement for upstream ``OmniParserClient`` (sync).

    Differences from the original:

    * Talks to ``/parse`` (not ``/parse/`` — trailing slash dropped) via
      :class:`omniparser_client.OmniparserClient`.
    * Returns a :class:`ParsedScreen` (a TypedDict) instead of a raw dict.
      Wire-level fields are unchanged.
    * Errors surface as :class:`omniparser_client.OmniparserHTTPError` /
      :class:`omniparser_client.OmniparserSchemaError` instead of bare
      ``requests`` exceptions.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float | None = None,
        transport: httpx.BaseTransport | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        kwargs: dict[str, object] = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        if transport is not None:
            kwargs["transport"] = transport
        if headers is not None:
            kwargs["headers"] = headers
        self._client = OmniparserClient(base_url, **kwargs)  # type: ignore[arg-type]

    def parse_screenshot(self, image: bytes | str | Path) -> ParsedScreen:
        """Parse a screenshot and return the legacy-shaped payload."""
        if isinstance(image, (bytes, bytearray)):
            image_b64 = base64.b64encode(bytes(image)).decode("ascii")
        elif isinstance(image, Path):
            image_b64 = base64.b64encode(image.read_bytes()).decode("ascii")
        else:
            image_b64 = image  # caller passed base64 already
        resp = self._client.parse(image_b64)
        return _adapt_response(resp, image_b64)

    def close(self) -> None:
        """Release underlying HTTP resources."""
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
