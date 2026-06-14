"""Errors raised by ``omnitool-agent``."""

from __future__ import annotations


class OmnitoolAgentError(Exception):
    """Base class for all ``omnitool-agent`` errors."""


class ImageDecodeError(OmnitoolAgentError):
    """Raised when an input image cannot be decoded to determine its size.

    The legacy contract requires ``width`` and ``height`` in the returned
    payload, so we need to decode the screenshot at adapter time.
    """
