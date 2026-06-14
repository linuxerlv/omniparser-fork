"""Public surface of the ``omnitool-agent`` package."""

from __future__ import annotations

from omnitool_agent.client_adapter import OmnitoolParserClient
from omnitool_agent.errors import ImageDecodeError, OmnitoolAgentError
from omnitool_agent.types import LegacyParsedElement, ParsedScreen

__all__ = (
    "ImageDecodeError",
    "LegacyParsedElement",
    "OmnitoolAgentError",
    "OmnitoolParserClient",
    "ParsedScreen",
)
