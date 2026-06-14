"""Tests for the typed ``ServerSettings`` env loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from omniparser_server.settings import ServerSettings


def test_defaults_safe_for_dev() -> None:
    s = ServerSettings()
    assert s.host == "127.0.0.1"
    assert s.port == 8000
    assert s.log_format == "json"
    assert s.eager_load is False
    assert s.detector_weights is None
    assert s.captioner_weights is None
    assert s.cors_origins == ()


def test_env_prefix_loads_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIPARSER_HOST", "0.0.0.0")
    monkeypatch.setenv("OMNIPARSER_PORT", "9001")
    monkeypatch.setenv("OMNIPARSER_LOG_FORMAT", "console")
    monkeypatch.setenv("OMNIPARSER_EAGER_LOAD", "true")
    s = ServerSettings()
    assert s.host == "0.0.0.0"
    assert s.port == 9001
    assert s.log_format == "console"
    assert s.eager_load is True


def test_path_fields_expand_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"")
    monkeypatch.setenv("OMNIPARSER_DETECTOR_WEIGHTS", str(weights))
    s = ServerSettings()
    assert s.detector_weights == weights.resolve()


def test_invalid_port_rejected() -> None:
    with pytest.raises(ValueError, match="port"):
        ServerSettings(port=0)


def test_settings_frozen() -> None:
    s = ServerSettings()
    with pytest.raises(ValueError, match="frozen"):
        s.host = "broken"  # type: ignore[misc]
