"""Small dependency-free lossless grayscale PNG reader/writer for mask artifacts."""

from __future__ import annotations

import binascii
import hashlib
import struct
import zlib
from pathlib import Path

from .models import MaskArtifact

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_PIXELS = 64_000_000


class MaskArtifactError(ValueError):
    """A mask artifact is not a bounded supported PNG."""


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
    )


def write_mask_png(path: Path, width: int, height: int, alpha: bytes) -> MaskArtifact:
    if not isinstance(path, Path) or not path.is_absolute():
        raise MaskArtifactError("mask path must be absolute")
    if not 1 <= width <= 16_384 or not 1 <= height <= 16_384 or width * height > _MAX_PIXELS:
        raise MaskArtifactError("mask dimensions exceed the safety limit")
    if len(alpha) != width * height:
        raise MaskArtifactError("mask alpha length does not match dimensions")
    rows = b"".join(b"\0" + alpha[y * width : (y + 1) * width] for y in range(height))
    payload = _SIGNATURE + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    payload += _chunk(b"IDAT", zlib.compress(rows, 6)) + _chunk(b"IEND", b"")
    path.write_bytes(payload)
    return validate_mask_png(path)


def write_rgba_mask_png(path: Path, width: int, height: int, alpha: bytes) -> Path:
    """Write a temporary RGBA bridge image whose alpha is the mask.

    GIMP's ``CHANNEL-OP-REPLACE`` selection path reads layer alpha.  Provider
    artifacts remain grayscale; this short-lived bridge avoids per-pixel PDB
    calls while retaining soft alpha.
    """

    if not isinstance(path, Path) or not path.is_absolute():
        raise MaskArtifactError("mask path must be absolute")
    if not 1 <= width <= 16_384 or not 1 <= height <= 16_384 or width * height > _MAX_PIXELS:
        raise MaskArtifactError("mask dimensions exceed the safety limit")
    if len(alpha) != width * height:
        raise MaskArtifactError("mask alpha length does not match dimensions")
    rows = b"".join(
        b"\0"
        + b"".join(bytes((255, 255, 255, value)) for value in alpha[y * width : (y + 1) * width])
        for y in range(height)
    )
    payload = _SIGNATURE + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    payload += _chunk(b"IDAT", zlib.compress(rows, 6)) + _chunk(b"IEND", b"")
    path.write_bytes(payload)
    return path


def _read_chunks(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    if len(data) > 128 * 1024 * 1024 or not data.startswith(_SIGNATURE):
        raise MaskArtifactError("mask is not a supported PNG")
    position = len(_SIGNATURE)
    width = height = None
    compressed = bytearray()
    while position + 12 <= len(data):
        length = struct.unpack(">I", data[position : position + 4])[0]
        position += 4
        if length > len(data) - position - 8:
            raise MaskArtifactError("mask PNG chunk exceeds file bounds")
        kind = data[position : position + 4]
        body = data[position + 4 : position + 4 + length]
        checksum = struct.unpack(">I", data[position + 4 + length : position + 8 + length])[0]
        if binascii.crc32(kind + body) & 0xFFFFFFFF != checksum:
            raise MaskArtifactError("mask PNG has an invalid CRC")
        position += length + 8
        if kind == b"IHDR":
            if len(body) != 13:
                raise MaskArtifactError("mask PNG has a malformed IHDR")
            width, height, depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", body
            )
            if depth != 8 or color_type != 0 or compression or filtering or interlace:
                raise MaskArtifactError("mask PNG must be non-interlaced 8-bit grayscale")
            if (
                not 1 <= width <= 16_384
                or not 1 <= height <= 16_384
                or width * height > _MAX_PIXELS
            ):
                raise MaskArtifactError("mask dimensions exceed the safety limit")
        elif kind == b"IDAT":
            compressed.extend(body)
        elif kind == b"IEND":
            break
    if width is None or height is None or not compressed:
        raise MaskArtifactError("mask PNG lacks required image data")
    try:
        raw = zlib.decompress(bytes(compressed))
    except zlib.error as exc:
        raise MaskArtifactError("mask PNG image data is not valid") from exc
    stride = width
    if len(raw) != (stride + 1) * height:
        raise MaskArtifactError("mask PNG data length does not match dimensions")
    pixels = bytearray()
    previous = bytearray(stride)
    offset = 0
    for _ in range(height):
        filter_type = raw[offset]
        row = bytearray(raw[offset + 1 : offset + 1 + stride])
        offset += stride + 1
        for index, value in enumerate(row):
            left = row[index - 1] if index else 0
            above = previous[index]
            upper_left = previous[index - 1] if index else 0
            if filter_type == 1:
                row[index] = (value + left) & 255
            elif filter_type == 2:
                row[index] = (value + above) & 255
            elif filter_type == 3:
                row[index] = (value + ((left + above) // 2)) & 255
            elif filter_type == 4:
                predictor = left + above - upper_left
                distances = (
                    abs(predictor - left),
                    abs(predictor - above),
                    abs(predictor - upper_left),
                )
                row[index] = (
                    value
                    + (
                        left
                        if distances[0] <= distances[1] and distances[0] <= distances[2]
                        else above
                        if distances[1] <= distances[2]
                        else upper_left
                    )
                ) & 255
            elif filter_type != 0:
                raise MaskArtifactError("mask PNG uses an unsupported row filter")
        pixels.extend(row)
        previous = row
    return width, height, bytes(pixels)


def validate_mask_png(path: Path) -> MaskArtifact:
    width, height, pixels = _read_chunks(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return MaskArtifact(
        path,
        width,
        height,
        soft_alpha=any(0 < value < 255 for value in pixels),
        sha256=digest,
    )


def read_mask_png(path: Path) -> tuple[MaskArtifact, bytes]:
    width, height, pixels = _read_chunks(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return MaskArtifact(
        path, width, height, soft_alpha=any(0 < value < 255 for value in pixels), sha256=digest
    ), pixels
