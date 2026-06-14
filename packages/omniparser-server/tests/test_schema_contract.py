"""Wire-shape contract test between server and client schemas.

The two packages keep separate copies of the schema models so that
``omniparser-client`` does not pull FastAPI / uvicorn / pydantic-settings into
environments that only want to talk to a remote parser. This test asserts
that the **wire-relevant** parts of every model's JSON Schema are identical
between the two packages, so any divergence is caught in CI rather than at
runtime.

Docstrings (``description`` fields) and titles are intentionally not part of
the contract: they are documentation, not protocol.
"""

from __future__ import annotations

from typing import Any

import pytest

from omniparser_client import schemas as client_schemas
from omniparser_server import schemas as server_schemas

# Pairs of (server-side, client-side) models that must stay isomorphic.
# Names live in this list rather than being discovered dynamically so that
# *deletions* on one side break the test loudly.
_PAIRS = (
    ("ParseRequest", server_schemas.ParseRequest, client_schemas.ParseRequest),
    ("ParseResponse", server_schemas.ParseResponse, client_schemas.ParseResponse),
    (
        "ParsedElementModel",
        server_schemas.ParsedElementModel,
        client_schemas.ParsedElementModel,
    ),
    ("HealthResponse", server_schemas.HealthResponse, client_schemas.HealthResponse),
    ("ErrorResponse", server_schemas.ErrorResponse, client_schemas.ErrorResponse),
)

# Keys that are pure documentation; differences are not protocol-breaking.
_DOC_KEYS = frozenset({"description", "title", "examples"})


def _strip_doc_keys(node: Any) -> Any:
    """Recursively drop documentation-only keys from a JSON Schema fragment."""

    if isinstance(node, dict):
        return {k: _strip_doc_keys(v) for k, v in node.items() if k not in _DOC_KEYS}
    if isinstance(node, list):
        return [_strip_doc_keys(item) for item in node]
    return node


@pytest.mark.parametrize(("name", "server_model", "client_model"), _PAIRS)
def test_schema_contract(name: str, server_model: type, client_model: type) -> None:
    server_schema = _strip_doc_keys(server_model.model_json_schema())  # type: ignore[attr-defined]
    client_schema = _strip_doc_keys(client_model.model_json_schema())  # type: ignore[attr-defined]
    assert server_schema == client_schema, f"wire schema drift detected on {name}"


def test_schema_version_aligned() -> None:
    assert server_schemas.SCHEMA_VERSION == client_schemas.SCHEMA_VERSION


def test_request_field_set_aligned() -> None:
    server_fields = set(server_schemas.ParseRequest.model_fields)
    client_fields = set(client_schemas.ParseRequest.model_fields)
    assert server_fields == client_fields


def test_response_field_set_aligned() -> None:
    server_fields = set(server_schemas.ParseResponse.model_fields)
    client_fields = set(client_schemas.ParseResponse.model_fields)
    assert server_fields == client_fields
