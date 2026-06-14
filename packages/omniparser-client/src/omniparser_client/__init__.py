"""Public surface of the ``omniparser-client`` package."""

from __future__ import annotations

from omniparser_client.client import (
    DEFAULT_TIMEOUT,
    AsyncOmniparserClient,
    OmniparserClient,
    encode_image,
)
from omniparser_client.errors import (
    OmniparserClientError,
    OmniparserHTTPError,
    OmniparserSchemaError,
)
from omniparser_client.schemas import (
    SCHEMA_VERSION,
    ErrorResponse,
    HealthResponse,
    ParsedElementModel,
    ParseRequest,
    ParseResponse,
)

__all__ = (
    "DEFAULT_TIMEOUT",
    "SCHEMA_VERSION",
    "AsyncOmniparserClient",
    "ErrorResponse",
    "HealthResponse",
    "OmniparserClient",
    "OmniparserClientError",
    "OmniparserHTTPError",
    "OmniparserSchemaError",
    "ParseRequest",
    "ParseResponse",
    "ParsedElementModel",
    "encode_image",
)
