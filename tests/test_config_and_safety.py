from __future__ import annotations

from pathlib import Path

import pytest

from gimp_local_mcp.config import Config
from gimp_local_mcp.errors import ConfigurationError, PathPolicyError, ScriptFuError
from gimp_local_mcp.service import GimpService


def test_default_config_is_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GIMP_MCP_HOST", raising=False)
    monkeypatch.delenv("GIMP_MCP_PORT", raising=False)
    config = Config.from_env()
    config.validate()
    assert config.host == "127.0.0.1"
    assert config.port == 10008
    assert config.allow_remote is False


def test_remote_host_requires_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIMP_MCP_HOST", "192.0.2.1")
    with pytest.raises(ConfigurationError):
        Config.from_env().validate()
    monkeypatch.setenv("GIMP_MCP_ALLOW_REMOTE", "true")
    Config.from_env().validate()


def test_output_policy_requires_existing_parent_and_explicit_overwrite(tmp_path: Path) -> None:
    existing = tmp_path / "existing.png"
    existing.write_bytes(b"not touched")
    with pytest.raises(PathPolicyError):
        GimpService._output_file(str(existing), overwrite=False)
    assert GimpService._output_file(str(existing), overwrite=True) == existing.resolve()
    with pytest.raises(PathPolicyError):
        GimpService._output_file(str(tmp_path / "missing" / "new.png"), overwrite=True)


def test_export_does_not_fall_back_to_save(tmp_path: Path) -> None:
    service = GimpService.__new__(GimpService)
    with pytest.raises(ScriptFuError, match="safe gimp-file-export"):
        service._export(5, tmp_path / "preview.png")


def test_vision_configuration_is_optional_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIMP_MCP_VISION_PROVIDER", "sam3")
    monkeypatch.setenv("GIMP_MCP_VISION_COMMAND", "python tools/vision/sam3_worker.py")
    config = Config.from_env()
    config.validate()
    assert config.vision_provider == "sam3"
    assert config.vision_command == ("python", "tools/vision/sam3_worker.py")


def test_vision_command_requires_a_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GIMP_MCP_VISION_PROVIDER", raising=False)
    monkeypatch.setenv("GIMP_MCP_VISION_COMMAND", "python worker.py")
    with pytest.raises(ConfigurationError, match="requires"):
        Config.from_env().validate()
