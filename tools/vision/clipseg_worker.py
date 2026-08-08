#!/usr/bin/env python3
"""Local CLIPSeg text-segmentation worker with offline-by-default model loading.

Install ML dependencies and download the checkpoint explicitly in a separate
environment. Normal JSONL operation uses only local files and never performs a
network request.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPOSITORY_SRC) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_SRC))

PROTOCOL_VERSION = "0.1"
MODEL_ID = os.getenv("GIMP_MCP_CLIPSEG_MODEL", "CIDAS/clipseg-rd64-refined")
DEVICE_REQUEST = os.getenv("GIMP_MCP_VISION_DEVICE", "auto").strip().lower()
MASK_THRESHOLD = 0.2
MASK_SLOPE = 2.0
MIN_AUTO_CUDA_FREE = 1536 * 1024 * 1024
logger = logging.getLogger("gimp-local-mcp.clipseg-worker")


class WorkerUnavailable(RuntimeError):
    pass


class WorkerOutOfMemory(RuntimeError):
    pass


class ClipSegRuntime:
    def __init__(self) -> None:
        self.torch: Any = None
        self.processor: Any = None
        self.model: Any = None
        self.device = "cpu"
        self.dtype = "float32"
        self.load_seconds: float | None = None
        self.self_test_seconds: float | None = None
        self.self_test_success = False
        self.reason: str | None = None
        self._diagnostics: dict[str, Any] = {}

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

    def _device_diagnostics(self, torch: Any) -> dict[str, Any]:
        cuda_available = bool(torch.cuda.is_available())
        result: dict[str, Any] = {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
            "cuda_available": cuda_available,
            "gpu_name": None,
            "compute_capability": None,
            "total_vram_bytes": None,
            "available_vram_bytes": None,
        }
        if cuda_available:
            free, total = torch.cuda.mem_get_info(0)
            major, minor = torch.cuda.get_device_capability(0)
            result.update(
                gpu_name=str(torch.cuda.get_device_name(0)),
                compute_capability=f"{major}.{minor}",
                total_vram_bytes=int(total),
                available_vram_bytes=int(free),
            )
        return result

    def _choose_device(self, diagnostics: dict[str, Any]) -> str:
        if DEVICE_REQUEST not in {"auto", "cpu", "cuda"}:
            raise WorkerUnavailable("GIMP_MCP_VISION_DEVICE must be auto, cpu, or cuda")
        if DEVICE_REQUEST == "cpu":
            return "cpu"
        if DEVICE_REQUEST == "cuda":
            if not diagnostics["cuda_available"]:
                raise WorkerUnavailable("CUDA was requested but is unavailable to provider Python")
            return "cuda"
        free = diagnostics.get("available_vram_bytes")
        if diagnostics["cuda_available"] and isinstance(free, int) and free >= MIN_AUTO_CUDA_FREE:
            return "cuda"
        return "cpu"

    def ensure_loaded(self) -> None:
        if self.model is not None:
            return
        started = time.perf_counter()
        torch, image_type, model_type, processor_type = self._import_dependencies()
        diagnostics = self._device_diagnostics(torch)
        device = self._choose_device(diagnostics)
        dtype = torch.float16 if device == "cuda" else torch.float32
        try:
            processor = processor_type.from_pretrained(MODEL_ID, local_files_only=True)
            model = model_type.from_pretrained(
                MODEL_ID,
                local_files_only=True,
                dtype=dtype,
            )
            model.to(device)
            model.eval()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise WorkerOutOfMemory(
                    "CLIPSeg model load exceeded available accelerator memory; use cpu"
                ) from exc
            raise WorkerUnavailable(f"CLIPSeg model load failed: {type(exc).__name__}") from exc
        except OSError as exc:
            raise WorkerUnavailable(
                "CLIPSeg checkpoint is not available locally; run this worker with "
                "--download-model during explicit setup"
            ) from exc
        self.torch = torch
        self.processor = processor
        self.model = model
        self.device = device
        self.dtype = str(dtype).removeprefix("torch.")
        self.load_seconds = time.perf_counter() - started
        self._diagnostics = diagnostics
        self._diagnostics["model_memory_bytes"] = int(
            sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        )
        self._self_test(image_type)

    def _self_test(self, image_type: Any) -> None:
        started = time.perf_counter()
        sample = image_type.new("RGB", (32, 32), (127, 127, 127))
        try:
            inputs = self.processor(
                text=["object"], images=[sample], padding=True, return_tensors="pt"
            )
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            with self.torch.inference_mode():
                output = self.model(**inputs).logits
            if output.numel() == 0 or not bool(self.torch.isfinite(output).all()):
                raise WorkerUnavailable("CLIPSeg self-test returned malformed logits")
            self.self_test_success = True
            self.self_test_seconds = time.perf_counter() - started
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
                raise WorkerOutOfMemory("CLIPSeg self-test exceeded accelerator memory") from exc
            raise WorkerUnavailable(f"CLIPSeg self-test failed: {type(exc).__name__}") from exc

    def capabilities(self) -> dict[str, Any]:
        try:
            self.ensure_loaded()
            available = True
            reason = None
        except (WorkerUnavailable, WorkerOutOfMemory) as exc:
            available = False
            reason = str(exc)
            self.reason = reason
            if self.torch is None:
                try:
                    torch, _image, _model, _processor = self._import_dependencies()
                    self.torch = torch
                    self._diagnostics = self._device_diagnostics(torch)
                except WorkerUnavailable:
                    pass
        return {
            "provider": "clipseg",
            "available": available,
            "model": MODEL_ID,
            "text_segmentation": available,
            "visual_prompts": False,
            "soft_alpha": True,
            "backend": "transformers-clipseg",
            "runtime": f"Python {platform.python_version()}",
            "reason": reason,
            "device": self.device if available else None,
            "accelerator": "CUDA" if available and self.device == "cuda" else "CPU",
            "dtype": self.dtype if available else None,
            **self._diagnostics,
            "checkpoint_available": self.model is not None,
            "model_load_success": self.model is not None,
            "test_inference_success": self.self_test_success,
            "instance_segmentation": False,
            "automatic_discovery": False,
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
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        started = time.perf_counter()
        try:
            inputs = self.processor(
                text=[prompt.strip()], images=[image], padding=True, return_tensors="pt"
            )
            inputs = {name: value.to(self.device) for name, value in inputs.items()}
            if self.device == "cuda":
                self.torch.cuda.reset_peak_memory_stats(0)
            with self.torch.inference_mode():
                logits = self.model(**inputs).logits.unsqueeze(1)
            logits = self.torch.nn.functional.interpolate(
                logits,
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            pivot = math.log(MASK_THRESHOLD / (1 - MASK_THRESHOLD))
            alpha = self.torch.sigmoid((logits - pivot) * MASK_SLOPE).float().cpu()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
                raise WorkerOutOfMemory(
                    "CLIPSeg inference exceeded accelerator memory; configure cpu fallback"
                ) from exc
            raise
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
        peak_vram = int(self.torch.cuda.max_memory_allocated(0)) if self.device == "cuda" else None
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
                        "mask_threshold": MASK_THRESHOLD,
                        "mask_slope": MASK_SLOPE,
                        "peak_activation": float(alpha.max()),
                        "mean_activation": float(alpha.mean()),
                        "peak_vram_bytes": peak_vram,
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
            "provenance": {
                "provider": "clipseg",
                "model": MODEL_ID,
                "device": self.device,
                "dtype": self.dtype,
                "model_load_seconds": self.load_seconds,
                "self_test_seconds": self.self_test_seconds,
                "peak_vram_bytes": peak_vram,
                "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            },
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
    except WorkerUnavailable as exc:
        return _error(request_id, "unavailable", str(exc))
    except (OSError, ValueError) as exc:
        return _error(request_id, "invalid-request", str(exc))
    except Exception as exc:
        logger.exception("CLIPSeg worker request failed")
        return _error(request_id, "worker-error", f"CLIPSeg request failed: {type(exc).__name__}")


def _download_model() -> int:
    _torch, _image, model_type, processor_type = ClipSegRuntime()._import_dependencies()
    processor_type.from_pretrained(MODEL_ID)
    model_type.from_pretrained(MODEL_ID)
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
