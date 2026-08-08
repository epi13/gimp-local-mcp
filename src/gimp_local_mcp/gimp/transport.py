"""Persistent, reconnectable TCP client for GIMP's Script-Fu server."""

from __future__ import annotations

import logging
import socket
from contextlib import suppress

from ..config import Config
from ..errors import (
    GimpConnectionError,
    GimpTimeoutError,
    ProtocolError,
    ScriptFuError,
)
from .protocol import decode_response_header, encode_request

logger = logging.getLogger(__name__)


class ScriptFuClient:
    """One-client-at-a-time Script-Fu TCP connection.

    A failed request is not retried because a write may have reached GIMP before
    the connection failed. The socket is discarded and a later call reconnects.
    """

    def __init__(self, config: Config) -> None:
        config.validate()
        self.config = config
        self._socket: socket.socket | None = None

    def connect(self) -> None:
        if self._socket is not None:
            return
        try:
            self._socket = socket.create_connection(
                (self.config.host, self.config.port), timeout=self.config.timeout
            )
            self._socket.settimeout(self.config.timeout)
        except TimeoutError as exc:
            raise GimpTimeoutError(
                f"Timed out connecting to GIMP Script-Fu at {self.config.host}:{self.config.port}"
            ) from exc
        except OSError as exc:
            raise GimpConnectionError(
                f"Could not connect to GIMP Script-Fu at "
                f"{self.config.host}:{self.config.port}: {exc}"
            ) from exc

    def close(self) -> None:
        if self._socket is not None:
            with suppress(OSError):
                self._socket.shutdown(socket.SHUT_RDWR)
            with suppress(OSError):
                self._socket.close()
            self._socket = None

    def execute(self, expression: str) -> str:
        self.connect()
        assert self._socket is not None
        try:
            self._socket.sendall(encode_request(expression))
            header = self._recv_exact(4)
            error, length = decode_response_header(header)
            if length > self.config.max_response_bytes:
                raise ProtocolError(
                    f"GIMP response is {length} bytes, above configured limit "
                    f"{self.config.max_response_bytes}"
                )
            body = self._recv_exact(length)
            try:
                response = body.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ProtocolError("GIMP response is not UTF-8") from exc
            if error:
                raise ScriptFuError(
                    response or "GIMP Script-Fu returned an unspecified error",
                    expression=expression,
                )
            return response
        except ScriptFuError:
            raise
        except TimeoutError as exc:
            self.close()
            raise GimpTimeoutError("Timed out waiting for GIMP Script-Fu response") from exc
        except (OSError, ProtocolError, GimpConnectionError):
            self.close()
            raise

    def _recv_exact(self, length: int) -> bytes:
        assert self._socket is not None
        chunks: list[bytes] = []
        remaining = length
        while remaining:
            try:
                chunk = self._socket.recv(remaining)
            except TimeoutError as exc:
                raise GimpTimeoutError("Timed out receiving a Script-Fu frame") from exc
            if not chunk:
                raise GimpConnectionError("GIMP Script-Fu closed the connection mid-response")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def __enter__(self) -> ScriptFuClient:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
