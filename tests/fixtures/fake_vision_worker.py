"""Deterministic protocol worker used by ordinary unit tests."""

from __future__ import annotations

import json
import os
import struct
import sys
import time
from pathlib import Path

from gimp_local_mcp.vision.artifacts import write_mask_png

VERSION = "0.1"


def dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return 8, 8
    return struct.unpack(">II", data[16:24])


def response(request: dict[str, object]) -> dict[str, object]:
    if request.get("protocol_version") != VERSION:
        return {
            "protocol_version": VERSION,
            "type": "error",
            "code": "protocol",
            "error": "bad version",
        }
    if os.getenv("FAKE_VISION_MODE") == "malformed":
        return {"protocol_version": VERSION, "type": "not-a-capabilities-response"}
    if request.get("type") == "capabilities":
        return {
            "protocol_version": VERSION,
            "type": "capabilities_response",
            "request_id": request.get("request_id"),
            "capabilities": {
                "provider": "fake",
                "available": True,
                "model": "fake-mask-v1",
                "text_segmentation": True,
                "visual_prompts": True,
                "soft_alpha": True,
                "backend": "deterministic-test",
                "reason": None,
            },
        }
    if request.get("type") != "segment":
        return {
            "protocol_version": VERSION,
            "type": "error",
            "code": "protocol",
            "error": "bad type",
        }
    if os.getenv("FAKE_VISION_MODE") == "sleep":
        time.sleep(3)
    body = request.get("request")
    if not isinstance(body, dict):
        return {
            "protocol_version": VERSION,
            "type": "error",
            "code": "protocol",
            "error": "bad request",
        }
    path = Path(str(body["image_path"]))
    output = Path(str(body["output_directory"]))
    width, height = dimensions(path)
    mode = os.getenv("FAKE_VISION_MODE", "partial")
    pixels = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            if width // 4 <= x < (width * 3) // 4 and height // 4 <= y < (height * 3) // 4:
                pixels[y * width + x] = 128 if mode == "partial" and (x + y) % 3 == 0 else 255
    mask_path = output / "candidate-0.png"
    write_mask_png(mask_path, width, height, bytes(pixels))
    return {
        "protocol_version": VERSION,
        "type": "segmentation_response",
        "request_id": request.get("request_id"),
        "result": {
            "provider": "fake",
            "model": "fake-mask-v1",
            "candidates": [
                {
                    "candidate_id": "fake:0",
                    "concept": body.get("prompt"),
                    "score": 0.91,
                    "bounding_box": {
                        "x": width / 4,
                        "y": height / 4,
                        "width": width / 2,
                        "height": height / 2,
                    },
                    "mask": {
                        "path": str(mask_path),
                        "width": width,
                        "height": height,
                        "soft_alpha": True,
                        "encoding": "png",
                    },
                    "width": width,
                    "height": height,
                    "metadata": {"test": True},
                }
            ],
            "runtime_seconds": 0.001,
            "warnings": [],
            "provenance": {"provider": "fake"},
            "soft_alpha": True,
        },
    }


for line in sys.stdin:
    try:
        request = json.loads(line)
        if not isinstance(request, dict):
            raise ValueError
        print(json.dumps(response(request), separators=(",", ":")), flush=True)
    except Exception:
        print(
            json.dumps(
                {
                    "protocol_version": VERSION,
                    "type": "error",
                    "code": "protocol",
                    "error": "malformed request",
                }
            ),
            flush=True,
        )
