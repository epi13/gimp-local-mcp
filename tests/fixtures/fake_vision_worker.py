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
        unavailable = os.getenv("FAKE_VISION_MODE") == "unavailable"
        return {
            "protocol_version": VERSION,
            "type": "capabilities_response",
            "request_id": request.get("request_id"),
            "capabilities": {
                "provider": "fake",
                "available": not unavailable,
                "model": "fake-mask-v1",
                "text_segmentation": not unavailable,
                "visual_prompts": True,
                "soft_alpha": True,
                "backend": "deterministic-test",
                "reason": "fake provider unavailable" if unavailable else None,
                "device": "cpu",
                "accelerator": "CPU",
                "dtype": "float32",
                "torch_version": "fake-torch",
                "torch_cuda_version": None,
                "cuda_available": False,
                "checkpoint_available": not unavailable,
                "model_load_success": not unavailable,
                "test_inference_success": not unavailable,
                "instance_segmentation": True,
                "automatic_discovery": False,
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
    if os.getenv("FAKE_VISION_MODE") == "oom":
        return {
            "protocol_version": VERSION,
            "type": "error",
            "request_id": request.get("request_id"),
            "code": "oom",
            "error": "fake provider ran out of memory",
        }
    if os.getenv("FAKE_VISION_MODE") == "unavailable":
        return {
            "protocol_version": VERSION,
            "type": "error",
            "request_id": request.get("request_id"),
            "code": "unavailable",
            "error": "fake provider unavailable",
        }
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
    instance_count = max(1, min(8, int(os.getenv("FAKE_VISION_INSTANCES", "1"))))
    candidates = []
    for instance in range(instance_count):
        pixels = bytearray(width * height)
        offset = min(instance, max(0, width // 8))
        for y in range(height):
            for x in range(width):
                if (
                    width // 4 + offset <= x < (width * 3) // 4 + offset
                    and height // 4 <= y < (height * 3) // 4
                ):
                    pixels[y * width + x] = 128 if mode == "partial" and (x + y) % 3 == 0 else 255
        mask_path = output / f"candidate-{instance}.png"
        write_mask_png(mask_path, width, height, bytes(pixels))
        candidates.append(
            {
                "candidate_id": f"fake:{instance}",
                "concept": body.get("prompt"),
                "score": 0.91 - instance * 0.01,
                "bounding_box": {
                    "x": width / 4 + offset,
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
                "metadata": {"test": True, "instance": instance},
            }
        )
    return {
        "protocol_version": VERSION,
        "type": "segmentation_response",
        "request_id": request.get("request_id"),
        "result": {
            "provider": "fake",
            "model": "fake-mask-v1",
            "candidates": candidates,
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
