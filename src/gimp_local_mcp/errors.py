"""Typed errors exposed by the bridge and its safety boundaries."""


class GimpMcpError(Exception):
    """Base class for expected, user-actionable errors."""


class ConfigurationError(GimpMcpError, ValueError):
    """The configured server or safety policy is invalid."""


class PathPolicyError(GimpMcpError, ValueError):
    """A path is missing, invalid, or violates an explicit safety rule."""


class ProcedureNameError(GimpMcpError, ValueError):
    """A PDB procedure identifier is not a valid GIMP name."""


class TransportError(GimpMcpError):
    """Base class for Script-Fu transport failures."""


class GimpConnectionError(TransportError):
    """The Script-Fu server could not be reached."""


class GimpTimeoutError(TransportError, TimeoutError):
    """The Script-Fu server did not respond before the configured timeout."""


class ProtocolError(TransportError):
    """A Script-Fu frame was malformed or exceeded configured limits."""


class ScriptFuError(TransportError):
    """GIMP evaluated the request and returned an error response."""

    def __init__(self, message: str, *, expression: str | None = None) -> None:
        super().__init__(message)
        self.expression = expression


class SchemeParseError(GimpMcpError, ValueError):
    """Script-Fu returned a value that the small safe reader cannot parse."""


class UnsafeOperationError(GimpMcpError):
    """The requested operation would need an unsafe or unknown state assumption."""
