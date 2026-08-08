#!/usr/bin/env python3
"""Optional SAM 3 worker entry point.

This file deliberately has no ML dependencies.  It is a protocol-safe
diagnostic worker until an operator installs a compatible SAM 3 environment and
adds the locally maintained adapter outside this core package.  It never
downloads weights and never sends image data over a network.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import platform
import sys

PROTOCOL_VERSION = "0.1"
logger = logging.getLogger("gimp-local-mcp.sam3-worker")


def _capabilities() -> dict[str, object]:
    package = importlib.util.find_spec("sam3")
    torch = importlib.util.find_spec("torch")
    reason = "SAM 3 is not installed in this worker environment"
    if package is not None and torch is not None:
        reason = (
            "SAM 3 packages are visible, but no repository-vendored adapter is selected; "
            "configure and audit an upstream-compatible local adapter separately"
        )
    return {
        "provider": "sam3",
        "available": False,
        "model": os.getenv("GIMP_MCP_SAM3_MODEL") or None,
        "text_segmentation": False,
        "visual_prompts": False,
        "soft_alpha": False,
        "backend": "torch-present" if torch is not None else "unavailable",
        "runtime": f"Python {platform.python_version()}",
        "reason": reason,
    }


def _response(request: dict[str, object]) -> dict[str, object]:
    request_type = request.get("type")
    if request.get("protocol_version") != PROTOCOL_VERSION:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "type": "error",
            "code": "protocol",
            "error": "unsupported protocol version",
        }
    if request_type == "capabilities":
        return {
            "protocol_version": PROTOCOL_VERSION,
            "type": "capabilities_response",
            "request_id": request.get("request_id", "invalid"),
            "capabilities": _capabilities(),
        }
    if request_type == "segment":
        return {
            "protocol_version": PROTOCOL_VERSION,
            "type": "error",
            "request_id": request.get("request_id", "invalid"),
            "code": "unavailable",
            "error": _capabilities()["reason"],
        }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": "error",
        "request_id": request.get("request_id", "invalid"),
        "code": "protocol",
        "error": "unsupported request type",
    }


def main() -> int:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(message)s")
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            response = _response(request)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("malformed provider request: %s", type(exc).__name__)
            response = {
                "protocol_version": PROTOCOL_VERSION,
                "type": "error",
                "code": "protocol",
                "error": "malformed request",
            }
        print(json.dumps(response, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
