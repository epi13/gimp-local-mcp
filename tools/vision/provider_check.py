#!/usr/bin/env python3
"""Bounded real-provider readiness and inference probe for local Forge evidence."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from gimp_local_mcp.config import Config
from gimp_local_mcp.vision.artifacts import write_mask_png
from gimp_local_mcp.vision.client import VisionClient
from gimp_local_mcp.vision.models import SegmentationRequest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capabilities-only", action="store_true")
    args = parser.parse_args()
    config = Config.from_env()
    config.validate()
    client = VisionClient(config)
    try:
        capabilities = client.capabilities()
        if not capabilities.available:
            print(
                json.dumps(
                    {
                        "provider_available": False,
                        "reason": capabilities.reason,
                        "capabilities": capabilities.as_dict(),
                    }
                )
            )
            return 2
        if args.capabilities_only:
            print(
                json.dumps(
                    {
                        "provider_available": True,
                        "capabilities": capabilities.as_dict(),
                    }
                )
            )
            return 0
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
            print(
                json.dumps(
                    {
                        "semantic_inference": bool(result.candidates),
                        "provider": result.provider,
                        "model": result.model,
                        "runtime_seconds": result.runtime_seconds,
                        "candidate_count": len(result.candidates),
                        "provenance": result.provenance,
                    }
                )
            )
            return 0 if result.candidates else 2
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
