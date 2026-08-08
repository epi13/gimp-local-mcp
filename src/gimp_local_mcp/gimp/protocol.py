"""Pure Script-Fu wire framing functions."""

from __future__ import annotations

import struct

from ..errors import ProtocolError

MAGIC = 0x47
REQUEST_HEADER_SIZE = 3
RESPONSE_HEADER_SIZE = 4
MAX_FRAME_LENGTH = 65_535


def encode_request(command: str) -> bytes:
    body = command.encode("utf-8")
    if len(body) > MAX_FRAME_LENGTH:
        raise ProtocolError("Script-Fu request exceeds the 16-bit protocol limit")
    return struct.pack(">BH", MAGIC, len(body)) + body


def decode_request(frame: bytes) -> str:
    if len(frame) < REQUEST_HEADER_SIZE:
        raise ProtocolError("Script-Fu request header is truncated")
    magic, length = struct.unpack(">BH", frame[:REQUEST_HEADER_SIZE])
    if magic != MAGIC:
        raise ProtocolError(f"Unexpected Script-Fu request magic byte: 0x{magic:02x}")
    body = frame[REQUEST_HEADER_SIZE:]
    if len(body) != length:
        raise ProtocolError(f"Request length says {length} bytes, received {len(body)}")
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("Script-Fu request is not UTF-8") from exc


def decode_response_header(header: bytes) -> tuple[bool, int]:
    if len(header) != RESPONSE_HEADER_SIZE:
        raise ProtocolError("Script-Fu response header is truncated")
    magic, error_flag, length = struct.unpack(">BBH", header)
    if magic != MAGIC:
        raise ProtocolError(f"Unexpected Script-Fu response magic byte: 0x{magic:02x}")
    if error_flag not in (0, 1):
        raise ProtocolError(f"Invalid Script-Fu response error flag: {error_flag}")
    return bool(error_flag), length
