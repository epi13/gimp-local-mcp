"""Strict JSONL protocol shared by the core and local vision workers."""

from __future__ import annotations

import json
import uuid
from typing import Any

from ..errors import GimpMcpError
from .models import SegmentationRequest, SegmentationResult, VisionCapabilities

PROTOCOL_VERSION = "0.1"
MAX_LINE_BYTES = 1_048_576


class VisionProtocolError(GimpMcpError, ValueError):
    """The worker returned a malformed or unsafe protocol message."""


class VisionUnavailableError(GimpMcpError, RuntimeError):
    """No configured local vision provider can answer the request."""


class VisionWorkerError(GimpMcpError, RuntimeError):
    """A configured worker failed without producing a valid response."""


class VisionOutOfMemoryError(VisionWorkerError):
    """The provider could not satisfy a request within accelerator memory."""


def _request_id() -> str:
    return uuid.uuid4().hex


def capabilities_request() -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": "capabilities",
        "request_id": _request_id(),
    }


def segmentation_request(request: SegmentationRequest) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": "segment",
        "request_id": _request_id(),
        "request": request.as_dict(),
    }


def encode_message(message: dict[str, Any]) -> bytes:
    if not isinstance(message, dict) or message.get("protocol_version") != PROTOCOL_VERSION:
        raise VisionProtocolError("message has an unsupported protocol version")
    try:
        encoded = (json.dumps(message, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
    except (TypeError, ValueError) as exc:
        raise VisionProtocolError("message is not JSON serializable") from exc
    if len(encoded) > MAX_LINE_BYTES:
        raise VisionProtocolError("protocol message exceeds the size limit")
    return encoded


def decode_message(line: bytes | str) -> dict[str, Any]:
    raw = line.encode() if isinstance(line, str) else line
    if len(raw) > MAX_LINE_BYTES:
        raise VisionProtocolError("protocol response exceeds the size limit")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VisionProtocolError("worker response is not valid JSON") from exc
    if not isinstance(value, dict) or value.get("protocol_version") != PROTOCOL_VERSION:
        raise VisionProtocolError("worker response has an unsupported protocol version")
    return value


def decode_capabilities(message: dict[str, Any]) -> VisionCapabilities:
    if message.get("type") == "error":
        raise VisionWorkerError(str(message.get("error", "vision worker error"))[:1024])
    if message.get("type") != "capabilities_response":
        raise VisionProtocolError("expected capabilities_response")
    return VisionCapabilities.from_dict(message.get("capabilities"))


def decode_segmentation(message: dict[str, Any]) -> SegmentationResult:
    if message.get("type") == "error":
        code = str(message.get("code", "worker-error"))
        error = str(message.get("error", "vision worker error"))[:1024]
        if code == "unavailable":
            raise VisionUnavailableError(error)
        if code == "oom":
            raise VisionOutOfMemoryError(error)
        raise VisionWorkerError(error)
    if message.get("type") != "segmentation_response":
        raise VisionProtocolError("expected segmentation_response")
    return SegmentationResult.from_dict(message.get("result"))
