#!/usr/bin/env python3
"""Bounded real-provider readiness and inference probe for local Forge evidence."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from gimp_local_mcp.config import Config
from gimp_local_mcp.vision.artifacts import read_mask_png, write_mask_png
from gimp_local_mcp.vision.client import VisionClient
from gimp_local_mcp.vision.models import SegmentationRequest

_BENCHMARK_MODES = {
    "cpu": ("cpu", "none"),
    "full-cuda": ("cuda", "none"),
    "sequential-cpu-offload": ("cuda", "sequential-cpu"),
    "auto": ("auto", "auto"),
}


def _probe(config: Config, *, capabilities_only: bool) -> tuple[int, dict[str, object]]:
    client = VisionClient(config)
    try:
        capabilities = client.capabilities()
        if not capabilities.available:
            return 2, {
                "provider_available": False,
                "reason": capabilities.reason,
                "capabilities": capabilities.as_dict(),
            }
        if capabilities_only:
            return 0, {
                "provider_available": True,
                "capabilities": capabilities.as_dict(),
            }
        with tempfile.TemporaryDirectory(prefix="gimp-mcp-provider-check-") as directory:
            root = Path(directory)
            image = write_mask_png(root / "input.png", 32, 32, bytes([127]) * 1024)
            result = client.segment(
                SegmentationRequest(
                    image.path,
                    prompt="object",
                    max_candidates=1,
                    output_directory=root,
                )
            )
            pixels = read_mask_png(result.candidates[0].mask.path)[1] if result.candidates else b""
            return (0 if result.candidates else 2), {
                "semantic_inference": bool(result.candidates),
                "provider": result.provider,
                "model": result.model,
                "runtime_seconds": result.runtime_seconds,
                "candidate_count": len(result.candidates),
                "mask_sha256": result.candidates[0].mask.sha256 if result.candidates else None,
                "mask_mean": sum(pixels) / len(pixels) if pixels else None,
                "_mask_pixels": pixels,
                "provenance": result.provenance,
                "capabilities": capabilities.as_dict(),
            }
    finally:
        client.close()


def _benchmark(mode_names: list[str]) -> int:
    original_device = os.environ.get("GIMP_MCP_VISION_DEVICE")
    original_offload = os.environ.get("GIMP_MCP_VISION_OFFLOAD")
    rows: list[dict[str, object]] = []
    masks: dict[str, bytes] = {}
    try:
        for name in mode_names:
            if name not in _BENCHMARK_MODES:
                raise ValueError(f"unsupported benchmark mode: {name}")
            device, offload = _BENCHMARK_MODES[name]
            os.environ["GIMP_MCP_VISION_DEVICE"] = device
            os.environ["GIMP_MCP_VISION_OFFLOAD"] = offload
            try:
                code, result = _probe(Config.from_env(), capabilities_only=False)
                pixels = result.pop("_mask_pixels", b"")
                if isinstance(pixels, bytes) and pixels:
                    masks[name] = pixels
                rows.append({"requested_mode": name, "exit_code": code, **result})
            except Exception as exc:
                rows.append(
                    {
                        "requested_mode": name,
                        "exit_code": 2,
                        "error": f"{type(exc).__name__}: {str(exc)[:512]}",
                    }
                )
    finally:
        for key, value in (
            ("GIMP_MCP_VISION_DEVICE", original_device),
            ("GIMP_MCP_VISION_OFFLOAD", original_offload),
        ):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    comparisons = []
    if masks:
        baseline_name = next(iter(masks))
        baseline = masks[baseline_name]
        for name, pixels in list(masks.items())[1:]:
            if len(pixels) == len(baseline):
                differences = [
                    abs(left - right) for left, right in zip(baseline, pixels, strict=True)
                ]
                comparisons.append(
                    {
                        "baseline": baseline_name,
                        "mode": name,
                        "mean_absolute_8bit_difference": sum(differences) / len(differences),
                        "maximum_8bit_difference": max(differences),
                    }
                )
    print(json.dumps({"benchmark_modes": rows, "mask_comparisons": comparisons}, indent=2))
    return 0 if any(row["exit_code"] == 0 for row in rows) else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capabilities-only", action="store_true")
    parser.add_argument(
        "--benchmark",
        default=None,
        metavar="MODES",
        help="comma-separated cpu,full-cuda,sequential-cpu-offload,auto comparison",
    )
    args = parser.parse_args()
    if args.benchmark:
        return _benchmark([item.strip() for item in args.benchmark.split(",") if item.strip()])
    config = Config.from_env()
    config.validate()
    code, result = _probe(config, capabilities_only=args.capabilities_only)
    result.pop("_mask_pixels", None)
    print(json.dumps(result))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
