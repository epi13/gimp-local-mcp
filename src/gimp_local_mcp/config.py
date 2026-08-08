"""Small environment-backed configuration with secure localhost defaults."""

from __future__ import annotations

import ipaddress
import logging
import math
import os
import shlex
from dataclasses import dataclass

from .errors import ConfigurationError


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean, got {value!r}")


@dataclass(frozen=True, slots=True)
class Config:
    """Runtime settings. Remote GIMP connections require explicit opt-in."""

    host: str = "127.0.0.1"
    port: int = 10008
    timeout: float = 10.0
    log_level: str = "INFO"
    max_response_bytes: int = 65_535
    allow_remote: bool = False
    vision_provider: str = "none"
    vision_command: tuple[str, ...] | None = None
    vision_timeout: float = 60.0
    vision_device: str = "auto"

    @classmethod
    def from_env(cls) -> Config:
        def integer(name: str, default: int, minimum: int, maximum: int) -> int:
            value = os.getenv(name)
            try:
                parsed = default if value is None else int(value)
            except ValueError as exc:
                raise ConfigurationError(f"{name} must be an integer, got {value!r}") from exc
            if not minimum <= parsed <= maximum:
                raise ConfigurationError(f"{name} must be between {minimum} and {maximum}")
            return parsed

        def positive_float(name: str, default: float) -> float:
            value = os.getenv(name)
            try:
                parsed = default if value is None else float(value)
            except ValueError as exc:
                raise ConfigurationError(f"{name} must be a number, got {value!r}") from exc
            if not math.isfinite(parsed) or parsed <= 0:
                raise ConfigurationError(f"{name} must be greater than zero")
            return parsed

        vision_provider = os.getenv("GIMP_MCP_VISION_PROVIDER", "none").strip().lower()
        if (
            not vision_provider
            or len(vision_provider) > 64
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
                for character in vision_provider
            )
        ):
            raise ConfigurationError("GIMP_MCP_VISION_PROVIDER contains unsupported characters")
        command_text = os.getenv("GIMP_MCP_VISION_COMMAND")
        vision_command: tuple[str, ...] | None = None
        if command_text and command_text.strip():
            try:
                parts = tuple(shlex.split(command_text))
            except ValueError as exc:
                raise ConfigurationError(
                    "GIMP_MCP_VISION_COMMAND is not valid shell-like syntax"
                ) from exc
            if not parts or len(parts) > 32 or any(len(part) > 4096 for part in parts):
                raise ConfigurationError("GIMP_MCP_VISION_COMMAND is outside the safety bounds")
            vision_command = parts

        log_level = os.getenv("GIMP_MCP_LOG_LEVEL", "INFO").upper()
        if log_level not in logging.getLevelNamesMapping():
            raise ConfigurationError(
                f"GIMP_MCP_LOG_LEVEL is not a valid logging level: {log_level}"
            )
        vision_device = os.getenv("GIMP_MCP_VISION_DEVICE", "auto").strip().lower()
        if vision_device not in {"auto", "cpu", "cuda"}:
            raise ConfigurationError("GIMP_MCP_VISION_DEVICE must be auto, cpu, or cuda")
        return cls(
            host=os.getenv("GIMP_MCP_HOST", "127.0.0.1"),
            port=integer("GIMP_MCP_PORT", 10008, 1, 65_535),
            timeout=positive_float("GIMP_MCP_TIMEOUT", 10.0),
            log_level=log_level,
            max_response_bytes=integer("GIMP_MCP_MAX_RESPONSE_BYTES", 65_535, 1, 65_535),
            allow_remote=_env_bool("GIMP_MCP_ALLOW_REMOTE", False),
            vision_provider=vision_provider,
            vision_command=vision_command,
            vision_timeout=positive_float("GIMP_MCP_VISION_TIMEOUT", 60.0),
            vision_device=vision_device,
        )

    def validate(self) -> None:
        if not self.host.strip():
            raise ConfigurationError("GIMP host cannot be empty")
        if not self.allow_remote:
            if self.host.lower() not in {"localhost", "127.0.0.1", "::1"}:
                try:
                    is_loopback = ipaddress.ip_address(self.host).is_loopback
                except ValueError:
                    is_loopback = False
                if not is_loopback:
                    raise ConfigurationError(
                        "GIMP host is not loopback; set GIMP_MCP_ALLOW_REMOTE=true "
                        "for an explicit remote opt-in"
                    )
        if self.port < 1 or self.port > 65_535:
            raise ConfigurationError("GIMP port must be between 1 and 65535")
        if self.max_response_bytes > 65_535:
            raise ConfigurationError("Script-Fu response size cannot exceed 65535 bytes")
        if (
            not self.vision_provider
            or len(self.vision_provider) > 64
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
                for character in self.vision_provider
            )
        ):
            raise ConfigurationError("Vision provider name contains unsupported characters")
        if not math.isfinite(self.vision_timeout) or not 0 < self.vision_timeout <= 600:
            raise ConfigurationError("Vision worker timeout must be between zero and 600 seconds")
        if self.vision_device not in {"auto", "cpu", "cuda"}:
            raise ConfigurationError("Vision device must be auto, cpu, or cuda")
        if self.vision_provider == "none" and self.vision_command is not None:
            raise ConfigurationError(
                "GIMP_MCP_VISION_COMMAND requires a configured vision provider"
            )
        if self.vision_command is not None and (
            not self.vision_command
            or len(self.vision_command) > 32
            or any(
                not isinstance(part, str) or not part or len(part) > 4096
                for part in self.vision_command
            )
        ):
            raise ConfigurationError("GIMP_MCP_VISION_COMMAND is outside the safety bounds")
