"""Public surface of the ``omniparser-server`` package.

Tests and downstream apps should import from here, not from internal modules.
"""

from __future__ import annotations

from omniparser_server.app import create_app
from omniparser_server.schemas import (
    SCHEMA_VERSION,
    ErrorResponse,
    HealthResponse,
    ParsedElementModel,
    ParseRequest,
    ParseResponse,
)
from omniparser_server.settings import ServerSettings

__all__ = (
    "SCHEMA_VERSION",
    "ErrorResponse",
    "HealthResponse",
    "ParseRequest",
    "ParseResponse",
    "ParsedElementModel",
    "ServerSettings",
    "create_app",
)
