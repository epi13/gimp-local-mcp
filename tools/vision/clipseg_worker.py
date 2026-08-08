#!/usr/bin/env python3
"""Offline-by-default CLIPSeg worker with measured Torch placement."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_SRC = REPOSITORY_ROOT / "src"
for search_path in (REPOSITORY_SRC, REPOSITORY_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from tools.vision.runtime import (  # noqa: E402
    PlacementDecision,
    PlacementError,
    RuntimePolicy,
    apply_placement,
    collect_cuda_diagnostics,
    cuda_memory_snapshot,
    decide_placement,
    fallback_policy_after_oom,
    is_cuda_oom,
    offload_evidence,
    parameter_storage_bytes,
    reset_cuda_peaks,
    restore_model_to_cpu,
    rss_peak_bytes,
)

PROTOCOL_VERSION = "0.1"
MODEL_ID = os.getenv("GIMP_MCP_CLIPSEG_MODEL", "CIDAS/clipseg-rd64-refined")
MODEL_REVISION = os.getenv("GIMP_MCP_CLIPSEG_REVISION") or None
DEFAULT_MASK_THRESHOLD = 0.2
DEFAULT_MASK_SLOPE = 2.0
logger = logging.getLogger("gimp-local-mcp.clipseg-worker")
CLIPSEG_PRELOAD_MODULE_CLASSES = (
    "CLIPSegVisionEmbeddings",
    "CLIPSegTextEmbeddings",
)


class WorkerUnavailable(RuntimeError):
    pass


class WorkerOutOfMemory(RuntimeError):
    pass


def clipseg_mask_settings(environ: Mapping[str, str] | None = None) -> tuple[float, float]:
    """Read bounded CLIPSeg probability-to-alpha controls from trusted worker config."""

    values = os.environ if environ is None else environ

    def bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
        raw = values.get(name)
        if raw is None or not raw.strip():
            return default
        try:
            value = float(raw)
        except ValueError as exc:
            raise WorkerUnavailable(f"{name} must be a finite number") from exc
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise WorkerUnavailable(f"{name} must be between {minimum} and {maximum}")
        return value

    return (
        bounded_float("GIMP_MCP_CLIPSEG_MASK_THRESHOLD", DEFAULT_MASK_THRESHOLD, 0.01, 0.99),
        bounded_float("GIMP_MCP_CLIPSEG_MASK_SLOPE", DEFAULT_MASK_SLOPE, 0.25, 8.0),
    )


class ClipSegRuntime:
    """Own one model and move it among CPU, CUDA, and Accelerate offload modes."""

    def __init__(self) -> None:
        self.torch: Any = None
        self.processor: Any = None
        self.model: Any = None
        self.image_type: Any = None
        self.policy: RuntimePolicy | None = None
        self.decision: PlacementDecision | None = None
        self.diagnostics: dict[str, Any] = {}
        self.memory: dict[str, Any] = {}
        self.evidence: dict[str, Any] = {}
        self.load_seconds: float | None = None
        self.self_test_seconds: float | None = None
        self.self_test_success = False
        self.placement_attempts: list[str] = []
        self.oom_recovery_path: list[str] = []
        self.mask_threshold = DEFAULT_MASK_THRESHOLD
        self.mask_slope = DEFAULT_MASK_SLOPE

    def _import_dependencies(self) -> tuple[Any, Any, Any, Any]:
        try:
            import torch
            from PIL import Image
            from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
        except ImportError as exc:
            raise WorkerUnavailable(
                f"required local vision package is unavailable: {exc.name or type(exc).__name__}"
            ) from exc
        return torch, Image, CLIPSegForImageSegmentation, CLIPSegProcessor

    def _place(self, policy: RuntimePolicy) -> None:
        if self.model is None:
            raise WorkerUnavailable("CLIPSeg model is not loaded")
        if self.decision is not None:
            restore_model_to_cpu(self.model, self.torch)
        decision = decide_placement(
            policy,
            self._cuda_diagnostics,
            self.model_storage_bytes,
            sequential_offload_supported=True,
        )
        self.decision = decision
        try:
            self.model = apply_placement(
                self.model,
                self.torch,
                decision,
                preload_module_classes=CLIPSEG_PRELOAD_MODULE_CLASSES,
            )
        except Exception as exc:
            if is_cuda_oom(self.torch, exc):
                raise WorkerOutOfMemory(
                    f"CLIPSeg {decision.execution_mode} model placement exhausted CUDA memory"
                ) from exc
            if isinstance(exc, PlacementError):
                raise WorkerUnavailable(str(exc)) from exc
            raise
        self.model.eval()
        self.placement_attempts.append(decision.execution_mode)
        self.memory["after_model_placement"] = cuda_memory_snapshot(self.torch)
        self.evidence = offload_evidence(self.model, inference_completed=False)

    def _recover_oom(self) -> bool:
        assert self.policy is not None and self.decision is not None
        next_policy = fallback_policy_after_oom(
            self.policy,
            self.decision.execution_mode,
            sequential_offload_supported=True,
        )
        if next_policy is None:
            return False
        previous = self.decision.execution_mode
        restore_model_to_cpu(self.model, self.torch)
        self.decision = None
        self._place(next_policy)
        transition = f"{previous}->{self.decision.execution_mode}"
        self.oom_recovery_path.append(transition)
        return True

    def _forward(self, image: Any, prompt: str) -> Any:
        attempts = 0
        while True:
            assert self.decision is not None
            inputs = None
            try:
                inputs = self.processor(
                    text=[prompt], images=[image], padding=True, return_tensors="pt"
                )
                inputs = {
                    name: value.to(self.decision.execution_device) for name, value in inputs.items()
                }
                reset_cuda_peaks(self.torch)
                with self.torch.inference_mode():
                    logits = self.model(**inputs).logits
                self.memory["inference_peak"] = cuda_memory_snapshot(self.torch)
                self.evidence = offload_evidence(self.model, inference_completed=True)
                return logits
            except Exception as exc:
                if not is_cuda_oom(self.torch, exc):
                    raise
                del inputs
                attempts += 1
                if attempts > 2 or not self._recover_oom():
                    raise WorkerOutOfMemory(
                        f"CLIPSeg {self.decision.execution_mode} inference exhausted memory"
                    ) from exc

    def ensure_loaded(self) -> None:
        if self.model is not None:
            return
        started = time.perf_counter()
        torch, image_type, model_type, processor_type = self._import_dependencies()
        self.torch = torch
        self.image_type = image_type
        self.mask_threshold, self.mask_slope = clipseg_mask_settings()
        self.policy = RuntimePolicy.from_env()
        self._cuda_diagnostics = collect_cuda_diagnostics(torch)
        self.diagnostics = self._cuda_diagnostics.as_dict()
        try:
            processor = processor_type.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, local_files_only=True
            )
            model = model_type.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                local_files_only=True,
                dtype=torch.float32,
            )
            model.eval()
        except OSError as exc:
            raise WorkerUnavailable(
                "CLIPSeg checkpoint is not available locally; run this worker with "
                "--download-model during explicit setup"
            ) from exc
        except Exception as exc:
            raise WorkerUnavailable(
                f"CLIPSeg model load failed: {type(exc).__name__}: {str(exc)[:256]}"
            ) from exc
        self.processor = processor
        self.model = model
        self.model_storage_bytes = parameter_storage_bytes(model)
        self.diagnostics["model_memory_bytes"] = self.model_storage_bytes
        try:
            self._place(self.policy)
        except WorkerOutOfMemory:
            assert self.decision is not None
            fallback = fallback_policy_after_oom(
                self.policy,
                self.decision.execution_mode,
                sequential_offload_supported=True,
            )
            if fallback is None:
                raise
            self._place(fallback)
            self.oom_recovery_path.append(f"load->{self.decision.execution_mode}")
        self.load_seconds = time.perf_counter() - started
        self._self_test()

    def _self_test(self) -> None:
        started = time.perf_counter()
        sample = self.image_type.new("RGB", (32, 32), (127, 127, 127))
        try:
            output = self._forward(sample, "object")
            if output.numel() == 0 or not bool(self.torch.isfinite(output).all()):
                raise WorkerUnavailable("CLIPSeg self-test returned malformed logits")
            self.self_test_success = True
            self.self_test_seconds = time.perf_counter() - started
        except WorkerOutOfMemory:
            raise
        except Exception as exc:
            raise WorkerUnavailable(
                f"CLIPSeg self-test failed: {type(exc).__name__}: {str(exc)[:256]}"
            ) from exc

    def capabilities(self) -> dict[str, Any]:
        reason = None
        try:
            self.ensure_loaded()
            available = True
        except (WorkerUnavailable, WorkerOutOfMemory, PlacementError) as exc:
            available = False
            reason = str(exc)
            if self.torch is None:
                try:
                    torch, _image, _model, _processor = self._import_dependencies()
                    self.torch = torch
                    self._cuda_diagnostics = collect_cuda_diagnostics(torch)
                    self.diagnostics = self._cuda_diagnostics.as_dict()
                except WorkerUnavailable:
                    pass
        decision = self.decision.as_dict() if self.decision else {}
        peak = self.memory.get("inference_peak", {})
        return {
            "provider": "clipseg",
            "available": available,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "text_segmentation": available,
            "visual_prompts": False,
            "soft_alpha": True,
            "backend": "transformers-clipseg",
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
            "configured_gpu_reserve_bytes": decision.get("configured_gpu_reserve_bytes"),
            "effective_gpu_budget_bytes": decision.get("effective_gpu_budget_bytes"),
            "model_storage_bytes": getattr(self, "model_storage_bytes", None),
            "checkpoint_available": self.model is not None,
            "model_load_success": self.model is not None,
            "test_inference_success": self.self_test_success,
            "instance_segmentation": False,
            "automatic_discovery": False,
            "peak_cuda_allocated_bytes": peak.get("cuda_peak_allocated_bytes"),
            "peak_cuda_reserved_bytes": peak.get("cuda_peak_reserved_bytes"),
            **self.evidence,
            "rss_peak_bytes": rss_peak_bytes(),
            "model_load_seconds": self.load_seconds,
            "test_inference_seconds": self.self_test_seconds,
            "mask_threshold": self.mask_threshold,
            "mask_slope": self.mask_slope,
        }

    def segment(self, request: dict[str, Any]) -> dict[str, Any]:
        self.ensure_loaded()
        image_path = Path(str(request.get("image_path", "")))
        output_directory = Path(str(request.get("output_directory", "")))
        prompt = request.get("prompt")
        mode = request.get("mode", "semantic")
        if not image_path.is_absolute() or not image_path.is_file():
            raise ValueError("image_path must be an existing absolute local file")
        if not output_directory.is_absolute() or not output_directory.is_dir():
            raise ValueError("output_directory must be an existing absolute local directory")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 256:
            raise ValueError("prompt must be a non-empty string of at most 256 characters")
        if mode == "automatic":
            raise WorkerUnavailable("CLIPSeg does not support automatic object discovery")
        image = self.image_type.open(image_path).convert("RGB")
        started = time.perf_counter()
        logits = self._forward(image, prompt.strip()).unsqueeze(1)
        logits = self.torch.nn.functional.interpolate(
            logits,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        pivot = math.log(self.mask_threshold / (1 - self.mask_threshold))
        alpha = self.torch.sigmoid((logits - pivot) * self.mask_slope).float().cpu()
        pixels = (alpha * 255).clamp(0, 255).byte().numpy().tobytes()
        from gimp_local_mcp.vision.artifacts import write_mask_png

        mask_path = (output_directory / "clipseg-candidate-0.png").resolve()
        if output_directory.resolve() not in mask_path.parents:
            raise ValueError("mask output escaped the requested directory")
        artifact = write_mask_png(mask_path, image.width, image.height, pixels)
        foreground = self.torch.where(alpha >= 0.5)
        box = None
        if foreground[0].numel():
            ys, xs = foreground
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            box = {"x": x0, "y": y0, "width": x1 - x0 + 1, "height": y1 - y0 + 1}
        runtime = time.perf_counter() - started
        peak = self.memory.get("inference_peak", {})
        provenance = {
            "provider": "clipseg",
            "model": MODEL_ID,
            **(self.decision.as_dict() if self.decision else {}),
            **self.diagnostics,
            **self.evidence,
            "memory": self.memory,
            "model_load_seconds": self.load_seconds,
            "self_test_seconds": self.self_test_seconds,
            "inference_seconds": runtime,
            "peak_cuda_allocated_bytes": peak.get("cuda_peak_allocated_bytes"),
            "peak_cuda_reserved_bytes": peak.get("cuda_peak_reserved_bytes"),
            "rss_peak_bytes": rss_peak_bytes(),
            "placement_attempts": self.placement_attempts,
            "oom_recovery_path": self.oom_recovery_path,
        }
        return {
            "provider": "clipseg",
            "model": MODEL_ID,
            "candidates": [
                {
                    "candidate_id": "clipseg:0",
                    "concept": prompt.strip(),
                    "score": None,
                    "bounding_box": box,
                    "mask": artifact.as_dict(),
                    "width": image.width,
                    "height": image.height,
                    "metadata": {
                        "semantic_probability_mask": True,
                        "mask_threshold": self.mask_threshold,
                        "mask_slope": self.mask_slope,
                        "peak_activation": float(alpha.max()),
                        "mean_activation": float(alpha.mean()),
                    },
                }
            ],
            "runtime_seconds": runtime,
            "warnings": [
                "CLIPSeg output is a coarse semantic probability mask, not an alpha matte."
            ]
            + (
                ["Instance mode returns one semantic region; instances are not separated."]
                if mode == "instance"
                else []
            ),
            "provenance": provenance,
            "soft_alpha": True,
        }


def _error(request_id: object, code: str, message: str) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "type": "error",
        "request_id": request_id if isinstance(request_id, str) else "invalid",
        "code": code,
        "error": message[:1024],
    }


def _response(runtime: ClipSegRuntime, request: dict[str, Any]) -> dict[str, Any]:
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
            return _error(request_id, "protocol", "unsupported or malformed request")
        return {
            "protocol_version": PROTOCOL_VERSION,
            "type": "segmentation_response",
            "request_id": request_id,
            "result": runtime.segment(request["request"]),
        }
    except WorkerOutOfMemory as exc:
        return _error(request_id, "oom", str(exc))
    except (WorkerUnavailable, PlacementError) as exc:
        return _error(request_id, "unavailable", str(exc))
    except (OSError, ValueError) as exc:
        return _error(request_id, "invalid-request", str(exc))
    except Exception as exc:
        logger.exception("CLIPSeg worker request failed")
        return _error(request_id, "worker-error", f"CLIPSeg request failed: {type(exc).__name__}")


def _download_model() -> int:
    _torch, _image, model_type, processor_type = ClipSegRuntime()._import_dependencies()
    processor_type.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model_type.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    print(f"Downloaded CLIPSeg model into the local cache: {MODEL_ID}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local offline-by-default CLIPSeg worker")
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="explicitly download the configured public checkpoint, then exit",
    )
    args = parser.parse_args(argv)
    if args.download_model:
        return _download_model()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(message)s")
    runtime = ClipSegRuntime()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
            response = _response(runtime, request)
        except (ValueError, json.JSONDecodeError):
            response = _error("invalid", "protocol", "malformed request")
        print(json.dumps(response, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
