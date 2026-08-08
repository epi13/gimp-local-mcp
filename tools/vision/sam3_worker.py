#!/usr/bin/env python3
"""Official Meta SAM 3 image adapter with explicit, offline checkpoint use."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_SRC = REPOSITORY_ROOT / "src"
for search_path in (REPOSITORY_SRC, REPOSITORY_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from tools.vision.runtime import (  # noqa: E402
    PlacementError,
    RuntimePolicy,
    apply_placement,
    collect_cuda_diagnostics,
    cuda_memory_snapshot,
    decide_placement,
    is_cuda_oom,
    offload_evidence,
    parameter_storage_bytes,
    reset_cuda_peaks,
    rss_peak_bytes,
)

PROTOCOL_VERSION = "0.1"
MODEL_VERSION = os.getenv("GIMP_MCP_SAM3_VERSION", "sam3.1").strip().lower()
CHECKPOINT = os.getenv("GIMP_MCP_SAM3_CHECKPOINT")
logger = logging.getLogger("gimp-local-mcp.sam3-worker")


class Sam3Unavailable(RuntimeError):
    pass


class Sam3Runtime:
    def __init__(self) -> None:
        self.torch: Any = None
        self.model: Any = None
        self.processor_type: Any = None
        self.decision: Any = None
        self.diagnostics: dict[str, Any] = {}
        self.evidence: dict[str, Any] = {}
        self.memory: dict[str, Any] = {}
        self.load_seconds: float | None = None
        self.test_seconds: float | None = None
        self.text_tested = False
        self.box_tested = False

    @staticmethod
    def checkpoint_path() -> Path | None:
        if not CHECKPOINT:
            return None
        path = Path(CHECKPOINT).expanduser()
        return path.resolve() if path.is_file() else None

    def _diagnose_torch(self) -> None:
        if self.torch is not None:
            return
        try:
            import torch
        except ImportError:
            return
        self.torch = torch
        self._cuda_diagnostics = collect_cuda_diagnostics(torch)
        self.diagnostics = self._cuda_diagnostics.as_dict()

    def ensure_loaded(self) -> None:
        if self.model is not None:
            return
        self._diagnose_torch()
        if importlib.util.find_spec("sam3") is None:
            raise Sam3Unavailable("official Meta SAM 3 package is not installed")
        checkpoint = self.checkpoint_path()
        if checkpoint is None:
            raise Sam3Unavailable(
                "SAM 3 checkpoint is not configured as an existing local file; set "
                "GIMP_MCP_SAM3_CHECKPOINT after explicit gated checkpoint setup"
            )
        if self.torch is None:
            raise Sam3Unavailable("Torch is not installed in the SAM 3 provider environment")
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:
            raise Sam3Unavailable(
                f"official SAM 3 image API import failed: {type(exc).__name__}: {exc}"
            ) from exc
        started = time.perf_counter()
        try:
            model = build_sam3_image_model(
                device="cpu",
                checkpoint_path=str(checkpoint),
                load_from_HF=False,
                enable_segmentation=True,
                enable_inst_interactivity=False,
                compile=False,
            )
            storage = parameter_storage_bytes(model)
            policy = RuntimePolicy.from_env()
            decision = decide_placement(
                policy,
                self._cuda_diagnostics,
                storage,
                sequential_offload_supported=True,
                workspace_mib=int(os.getenv("GIMP_MCP_SAM3_WORKSPACE_MIB", "768")),
            )
            model = apply_placement(model, self.torch, decision)
            model.eval()
        except Exception as exc:
            if is_cuda_oom(self.torch, exc):
                raise Sam3Unavailable(
                    "SAM 3 model construction or placement exhausted available memory"
                ) from exc
            if isinstance(exc, PlacementError):
                raise Sam3Unavailable(str(exc)) from exc
            raise Sam3Unavailable(
                f"SAM 3 model load failed: {type(exc).__name__}: {str(exc)[:512]}"
            ) from exc
        self.model = model
        self.processor_type = Sam3Processor
        self.model_storage_bytes = storage
        self.decision = decision
        self.load_seconds = time.perf_counter() - started
        self.memory["after_model_placement"] = cuda_memory_snapshot(self.torch)
        self.evidence = offload_evidence(model, inference_completed=False)
        self._self_test()

    def _processor(self, threshold: float = 0.5) -> Any:
        return self.processor_type(
            self.model,
            device=self.decision.execution_device,
            confidence_threshold=threshold,
        )

    def _self_test(self) -> None:
        from PIL import Image

        sample = Image.new("RGB", (32, 32), (127, 127, 127))
        started = time.perf_counter()
        try:
            processor = self._processor()
            state = processor.set_image(sample)
            state = processor.set_text_prompt("object", state)
            self.text_tested = all(key in state for key in ("masks", "boxes", "scores"))
            state = processor.set_image(sample)
            state = processor.add_geometric_prompt([0.5, 0.5, 0.5, 0.5], True, state)
            self.box_tested = all(key in state for key in ("masks", "boxes", "scores"))
            self.evidence = offload_evidence(self.model, inference_completed=True)
            self.memory["self_test_peak"] = cuda_memory_snapshot(self.torch)
            self.test_seconds = time.perf_counter() - started
        except Exception as exc:
            if is_cuda_oom(self.torch, exc):
                raise Sam3Unavailable(
                    "SAM 3 loaded, but activation/workspace memory exhausted during inference"
                ) from exc
            raise Sam3Unavailable(
                f"SAM 3 real self-test failed: {type(exc).__name__}: {str(exc)[:512]}"
            ) from exc

    def capabilities(self) -> dict[str, Any]:
        reason = None
        try:
            self.ensure_loaded()
            available = self.text_tested or self.box_tested
        except Sam3Unavailable as exc:
            available = False
            reason = str(exc)
            self._diagnose_torch()
        decision = self.decision.as_dict() if self.decision else {}
        peak = self.memory.get("self_test_peak", {})
        return {
            "provider": "sam3",
            "available": available,
            "model": f"facebook/{MODEL_VERSION}",
            "model_revision": MODEL_VERSION,
            "text_segmentation": self.text_tested,
            "visual_prompts": self.box_tested,
            "soft_alpha": False,
            "backend": "official-meta-sam3",
            "runtime": f"Python {platform.python_version()}",
            "reason": reason,
            "device": decision.get("device"),
            "execution_device": decision.get("execution_device"),
            "execution_mode": decision.get("execution_mode"),
            "offload_mode": decision.get("offload_mode"),
            "placement_reason": decision.get("reason"),
            "accelerator": "CUDA" if decision.get("device") == "cuda" else "CPU",
            "dtype": decision.get("dtype"),
            **self.diagnostics,
            "checkpoint_available": self.checkpoint_path() is not None,
            "model_load_success": self.model is not None,
            "test_inference_success": self.text_tested,
            "instance_segmentation": self.text_tested,
            "automatic_discovery": False,
            "configured_gpu_reserve_bytes": decision.get("configured_gpu_reserve_bytes"),
            "effective_gpu_budget_bytes": decision.get("effective_gpu_budget_bytes"),
            "model_storage_bytes": getattr(self, "model_storage_bytes", None),
            "peak_cuda_allocated_bytes": peak.get("cuda_peak_allocated_bytes"),
            "peak_cuda_reserved_bytes": peak.get("cuda_peak_reserved_bytes"),
            **self.evidence,
            "rss_peak_bytes": rss_peak_bytes(),
            "model_load_seconds": self.load_seconds,
            "test_inference_seconds": self.test_seconds,
        }

    def segment(self, request: dict[str, Any]) -> dict[str, Any]:
        self.ensure_loaded()
        image_path = Path(str(request.get("image_path", "")))
        output_directory = Path(str(request.get("output_directory", "")))
        prompt = request.get("prompt")
        boxes = request.get("box_prompts", [])
        points = request.get("point_prompts", [])
        maximum = request.get("max_candidates", 3)
        minimum = request.get("minimum_score", 0.0)
        if not image_path.is_absolute() or not image_path.is_file():
            raise ValueError("image_path must be an existing absolute local file")
        if not output_directory.is_absolute() or not output_directory.is_dir():
            raise ValueError("output_directory must be an existing absolute local directory")
        if points:
            raise Sam3Unavailable(
                "the audited SAM 3 grounding adapter supports text and boxes; point prompts "
                "require the separate interactive predictor and are not yet exposed"
            )
        if not isinstance(boxes, list) or len(boxes) > 32:
            raise ValueError("box_prompts must be a bounded list")
        if prompt is not None and (
            not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 256
        ):
            raise ValueError("prompt must be null or a non-empty string up to 256 characters")
        if prompt is None and not boxes:
            raise ValueError("SAM 3 requires a text or box prompt")
        if not isinstance(maximum, int) or isinstance(maximum, bool) or not 1 <= maximum <= 32:
            raise ValueError("max_candidates must be between 1 and 32")
        if not isinstance(minimum, (int, float)) or not 0 <= float(minimum) <= 1:
            raise ValueError("minimum_score must be between 0 and 1")
        from PIL import Image

        from gimp_local_mcp.vision.artifacts import write_mask_png

        image = Image.open(image_path).convert("RGB")
        processor = self._processor(float(minimum))
        reset_cuda_peaks(self.torch)
        started = time.perf_counter()
        state = processor.set_image(image)
        if prompt is not None:
            state = processor.set_text_prompt(prompt.strip(), state)
        for box in boxes:
            if not isinstance(box, dict):
                raise ValueError("each box prompt must be an object")
            x, y = float(box["x"]), float(box["y"])
            width, height = float(box["width"]), float(box["height"])
            normalized = [
                (x + width / 2) / image.width,
                (y + height / 2) / image.height,
                width / image.width,
                height / image.height,
            ]
            if any(value < 0 or value > 1 for value in normalized):
                raise ValueError("box prompt must remain within the source image")
            state = processor.add_geometric_prompt(normalized, True, state)
        masks, result_boxes, scores = state["masks"], state["boxes"], state["scores"]
        order = self.torch.argsort(scores, descending=True)[:maximum]
        candidates = []
        for output_index, tensor_index in enumerate(order.tolist()):
            mask = masks[tensor_index].squeeze().to("cpu")
            pixels = (mask.to(self.torch.uint8) * 255).numpy().tobytes()
            path = (output_directory / f"sam3-candidate-{output_index}.png").resolve()
            if output_directory.resolve() not in path.parents:
                raise ValueError("mask output escaped the requested directory")
            artifact = write_mask_png(path, image.width, image.height, pixels)
            x0, y0, x1, y1 = [float(item) for item in result_boxes[tensor_index].tolist()]
            candidates.append(
                {
                    "candidate_id": f"sam3:{output_index}",
                    "concept": prompt.strip() if isinstance(prompt, str) else None,
                    "score": float(scores[tensor_index]),
                    "bounding_box": {
                        "x": max(0.0, x0),
                        "y": max(0.0, y0),
                        "width": max(1e-6, x1 - x0),
                        "height": max(1e-6, y1 - y0),
                    },
                    "mask": artifact.as_dict(),
                    "width": image.width,
                    "height": image.height,
                    "metadata": {"binary_segmentation_mask": True},
                }
            )
        runtime = time.perf_counter() - started
        self.memory["inference_peak"] = cuda_memory_snapshot(self.torch)
        self.evidence = offload_evidence(self.model, inference_completed=True)
        return {
            "provider": "sam3",
            "model": f"facebook/{MODEL_VERSION}",
            "candidates": candidates,
            "runtime_seconds": runtime,
            "warnings": [
                "SAM 3 masks are binary segmentations, not alpha mattes or fur/hair refinement."
            ],
            "provenance": {
                **self.decision.as_dict(),
                **self.diagnostics,
                **self.evidence,
                "memory": self.memory,
                "model_load_seconds": self.load_seconds,
                "inference_seconds": runtime,
                "rss_peak_bytes": rss_peak_bytes(),
            },
            "soft_alpha": False,
        }


def _error(request_id: object, code: str, message: object) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": "error",
        "request_id": request_id if isinstance(request_id, str) else "invalid",
        "code": code,
        "error": str(message)[:1024],
    }


def _response(runtime: Sam3Runtime, request: dict[str, Any]) -> dict[str, object]:
    request_id = request.get("request_id")
    if request.get("protocol_version") != PROTOCOL_VERSION:
        return _error(request_id, "protocol", "unsupported protocol version")
    try:
        if request.get("type") == "capabilities":
            return {
                "protocol_version": PROTOCOL_VERSION,
                "type": "capabilities_response",
                "request_id": request_id,
                "capabilities": runtime.capabilities(),
            }
        if request.get("type") != "segment" or not isinstance(request.get("request"), dict):
            return _error(request_id, "protocol", "unsupported request type")
        return {
            "protocol_version": PROTOCOL_VERSION,
            "type": "segmentation_response",
            "request_id": request_id,
            "result": runtime.segment(request["request"]),
        }
    except Sam3Unavailable as exc:
        return _error(request_id, "unavailable", exc)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return _error(request_id, "invalid-request", exc)
    except Exception as exc:
        logger.exception("SAM 3 worker request failed")
        return _error(request_id, "worker-error", f"SAM 3 request failed: {type(exc).__name__}")


def _download_checkpoint() -> int:
    from sam3.model_builder import download_ckpt_from_hf

    path = download_ckpt_from_hf(version=MODEL_VERSION)
    print(f"Downloaded gated SAM 3 checkpoint during explicit setup: {path}")
    print("Set GIMP_MCP_SAM3_CHECKPOINT to that local file before normal offline use.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Official offline-by-default SAM 3 worker")
    parser.add_argument(
        "--download-checkpoint",
        action="store_true",
        help="explicitly use authenticated Hugging Face access to fetch the gated checkpoint",
    )
    args = parser.parse_args(argv)
    if args.download_checkpoint:
        return _download_checkpoint()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(message)s")
    runtime = Sam3Runtime()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            response = _response(runtime, request)
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("malformed provider request: %s", type(exc).__name__)
            response = _error("invalid", "protocol", "malformed request")
        print(json.dumps(response, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
