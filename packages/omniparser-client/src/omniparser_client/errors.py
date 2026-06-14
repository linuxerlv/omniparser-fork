"""Typed exceptions raised by :class:`OmniparserClient`."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omniparser_client.schemas import ErrorResponse


class OmniparserClientError(Exception):
    """Base class for all client-side errors."""


class OmniparserHTTPError(OmniparserClientError):
    """Raised when the server returns a 4xx/5xx status."""

    def __init__(self, status_code: int, body: ErrorResponse | None, raw_text: str) -> None:
        self.status_code = status_code
        self.body = body
        self.raw_text = raw_text
        detail = body.detail if body is not None and body.detail is not None else raw_text
        super().__init__(f"HTTP {status_code}: {detail}")


class OmniparserSchemaError(OmniparserClientError):
    """Raised when the server response cannot be parsed against the wire schema.

    Most often this means the server has bumped its schema version while the
    client is still on the previous one.
    """
