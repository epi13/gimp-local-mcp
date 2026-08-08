"""Small environment-backed configuration with secure localhost defaults."""

from __future__ import annotations

import ipaddress
import logging
import os
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
            if parsed <= 0:
                raise ConfigurationError(f"{name} must be greater than zero")
            return parsed

        log_level = os.getenv("GIMP_MCP_LOG_LEVEL", "INFO").upper()
        if log_level not in logging.getLevelNamesMapping():
            raise ConfigurationError(
                f"GIMP_MCP_LOG_LEVEL is not a valid logging level: {log_level}"
            )
        return cls(
            host=os.getenv("GIMP_MCP_HOST", "127.0.0.1"),
            port=integer("GIMP_MCP_PORT", 10008, 1, 65_535),
            timeout=positive_float("GIMP_MCP_TIMEOUT", 10.0),
            log_level=log_level,
            max_response_bytes=integer("GIMP_MCP_MAX_RESPONSE_BYTES", 65_535, 1, 65_535),
            allow_remote=_env_bool("GIMP_MCP_ALLOW_REMOTE", False),
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
