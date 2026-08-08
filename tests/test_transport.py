from __future__ import annotations

import socket
import threading
from contextlib import closing

import pytest

from gimp_local_mcp.config import Config
from gimp_local_mcp.errors import GimpConnectionError, GimpTimeoutError, ScriptFuError
from gimp_local_mcp.gimp.protocol import decode_request
from gimp_local_mcp.gimp.transport import ScriptFuClient


def _receive(sock: socket.socket, amount: int) -> bytes:
    chunks: list[bytes] = []
    while len(b"".join(chunks)) < amount:
        part = sock.recv(amount - len(b"".join(chunks)))
        if not part:
            raise RuntimeError("client closed")
        chunks.append(part)
    return b"".join(chunks)


def _server(
    response: bytes, *, split: bool = False, close_early: bool = False
) -> tuple[int, threading.Thread]:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def run() -> None:
        with closing(listener):
            connection, _ = listener.accept()
            with closing(connection):
                header = _receive(connection, 3)
                length = int.from_bytes(header[1:], "big")
                request = _receive(connection, length)
                assert decode_request(header + request) == "(gimp-version)"
                if close_early:
                    return
                if split:
                    for byte in response:
                        connection.sendall(bytes([byte]))
                else:
                    connection.sendall(response)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return port, thread


def _config(port: int, timeout: float = 1.0) -> Config:
    return Config(host="127.0.0.1", port=port, timeout=timeout)


def test_transport_handles_partial_response_frames() -> None:
    body = b'"3.2.4"'
    response = b"G\x00" + len(body).to_bytes(2, "big") + body
    port, thread = _server(response, split=True)
    client = ScriptFuClient(_config(port))
    assert client.execute("(gimp-version)") == '"3.2.4"'
    client.close()
    thread.join(timeout=1)


def test_transport_surfaces_scriptfu_errors() -> None:
    body = b"unknown procedure"
    port, thread = _server(b"G\x01" + len(body).to_bytes(2, "big") + body)
    client = ScriptFuClient(_config(port))
    with pytest.raises(ScriptFuError, match="unknown procedure"):
        client.execute("(gimp-version)")
    client.close()
    thread.join(timeout=1)


def test_transport_surfaces_connection_close() -> None:
    port, thread = _server(b"", close_early=True)
    client = ScriptFuClient(_config(port))
    with pytest.raises(GimpConnectionError):
        client.execute("(gimp-version)")
    thread.join(timeout=1)


def test_transport_timeout_is_typed() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def run() -> None:
        connection, _ = listener.accept()
        with closing(connection):
            _receive(connection, 3)
            length = 0
            # The request body is not relevant; keep the peer waiting for a response.
            if length:
                _receive(connection, length)
            threading.Event().wait(0.2)
        listener.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    client = ScriptFuClient(_config(port, timeout=0.02))
    with pytest.raises(GimpTimeoutError):
        client.execute("(gimp-version)")
    thread.join(timeout=1)
