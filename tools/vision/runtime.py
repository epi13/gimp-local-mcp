"""Reusable Torch placement, sequential-offload, and memory diagnostics.

This module intentionally has no import-time Torch or Accelerate dependency. Vision
workers run it from separately managed provider environments; ordinary project tests
exercise the deterministic policy with simulated diagnostics.
"""

from __future__ import annotations

import gc
import os
import platform
import resource
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

MIB = 1024 * 1024
DEFAULT_WORKSPACE_MIB = 256


class PlacementError(RuntimeError):
    """The requested execution policy cannot be satisfied safely."""


@dataclass(frozen=True, slots=True)
class RuntimePolicy:
    device: str = "auto"
    offload: str = "auto"
    gpu_reserve_mib: int = 256
    max_vram_mib: int | None = None
    dtype: str = "auto"

    @classmethod
    def from_env(cls) -> RuntimePolicy:
        def integer(name: str, default: int, minimum: int) -> int:
            text = os.getenv(name)
            try:
                value = default if text is None else int(text)
            except ValueError as exc:
                raise PlacementError(f"{name} must be an integer") from exc
            if value < minimum:
                raise PlacementError(f"{name} must be at least {minimum}")
            return value

        device = os.getenv("GIMP_MCP_VISION_DEVICE", "auto").strip().lower()
        offload = os.getenv("GIMP_MCP_VISION_OFFLOAD", "auto").strip().lower()
        dtype = os.getenv("GIMP_MCP_VISION_DTYPE", "auto").strip().lower()
        maximum = os.getenv("GIMP_MCP_VISION_MAX_VRAM_MIB")
        policy = cls(
            device=device,
            offload=offload,
            gpu_reserve_mib=integer("GIMP_MCP_VISION_GPU_RESERVE_MIB", 256, 0),
            max_vram_mib=(
                integer("GIMP_MCP_VISION_MAX_VRAM_MIB", 128, 128) if maximum is not None else None
            ),
            dtype=dtype,
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise PlacementError("vision device must be auto, cpu, or cuda")
        if self.offload not in {"auto", "none", "sequential-cpu"}:
            raise PlacementError("vision offload must be auto, none, or sequential-cpu")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise PlacementError("vision dtype must be auto, float32, float16, or bfloat16")
        if self.device == "cpu" and self.offload == "sequential-cpu":
            raise PlacementError("sequential CPU offload requires auto or cuda device mode")
        if self.gpu_reserve_mib < 0:
            raise PlacementError("GPU reserve cannot be negative")
        if self.max_vram_mib is not None and self.max_vram_mib < 128:
            raise PlacementError("maximum VRAM must be at least 128 MiB")


@dataclass(frozen=True, slots=True)
class CudaDiagnostics:
    torch_version: str
    torch_cuda_version: str | None
    cuda_available: bool
    cuda_kernel_smoke_test_success: bool
    cuda_kernel_smoke_test_error: str | None = None
    gpu_name: str | None = None
    compute_capability: str | None = None
    torch_supported_architectures: tuple[str, ...] = ()
    total_vram_bytes: int | None = None
    free_vram_bytes_at_startup: int | None = None
    float16_execution_success: bool | None = None
    bfloat16_execution_success: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["torch_supported_architectures"] = list(self.torch_supported_architectures)
        result["available_vram_bytes"] = self.free_vram_bytes_at_startup
        return result


@dataclass(frozen=True, slots=True)
class PlacementDecision:
    execution_mode: str
    device: str
    execution_device: str
    offload_mode: str
    dtype: str
    reason: str
    configured_gpu_reserve_bytes: int
    configured_max_vram_bytes: int | None
    effective_gpu_budget_bytes: int
    estimated_model_bytes: int
    estimated_workspace_bytes: int
    full_cuda_required_bytes: int
    sequential_offload_supported: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:512]


def _probe_dtype(torch: Any, dtype: Any) -> tuple[bool, str | None]:
    try:
        left = torch.ones((32, 32), device="cuda", dtype=dtype)
        right = torch.ones((32, 32), device="cuda", dtype=dtype)
        output = left @ right
        torch.cuda.synchronize(0)
        success = bool(torch.isfinite(output).all().item())
        del left, right, output
        return success, None if success else "CUDA dtype probe returned non-finite values"
    except Exception as exc:
        return False, _error_text(exc)


def collect_cuda_diagnostics(torch: Any) -> CudaDiagnostics:
    """Probe CUDA discovery and actual kernel execution without trusting discovery alone."""

    cuda_available = bool(torch.cuda.is_available())
    base: dict[str, Any] = {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        "cuda_available": cuda_available,
        "cuda_kernel_smoke_test_success": False,
        "torch_supported_architectures": tuple(
            str(item) for item in (torch.cuda.get_arch_list() if cuda_available else [])
        ),
    }
    if not cuda_available:
        base["cuda_kernel_smoke_test_error"] = (
            "installed Torch build does not expose an available CUDA runtime"
        )
        return CudaDiagnostics(**base)
    try:
        free, total = torch.cuda.mem_get_info(0)
        major, minor = torch.cuda.get_device_capability(0)
        base.update(
            gpu_name=str(torch.cuda.get_device_name(0)),
            compute_capability=f"{major}.{minor}",
            total_vram_bytes=int(total),
            free_vram_bytes_at_startup=int(free),
        )
    except Exception as exc:
        base["cuda_kernel_smoke_test_error"] = _error_text(exc)
        return CudaDiagnostics(**base)
    success, error = _probe_dtype(torch, torch.float32)
    base["cuda_kernel_smoke_test_success"] = success
    base["cuda_kernel_smoke_test_error"] = error
    if success:
        base["float16_execution_success"] = _probe_dtype(torch, torch.float16)[0]
        bf16_reported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        base["bfloat16_execution_success"] = (
            _probe_dtype(torch, torch.bfloat16)[0] if bf16_reported else False
        )
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return CudaDiagnostics(**base)


def effective_gpu_budget_bytes(
    free_vram_bytes: int | None,
    reserve_mib: int,
    max_vram_mib: int | None,
) -> int:
    if free_vram_bytes is None:
        return 0
    capped = free_vram_bytes
    if max_vram_mib is not None:
        capped = min(capped, max_vram_mib * MIB)
    return max(0, capped - reserve_mib * MIB)


def _cuda_dtype(policy: RuntimePolicy, diagnostics: CudaDiagnostics) -> str:
    if policy.dtype == "float16":
        if diagnostics.float16_execution_success is not True:
            raise PlacementError("float16 was requested but its CUDA execution probe failed")
        return "float16"
    if policy.dtype == "bfloat16":
        if diagnostics.bfloat16_execution_success is not True:
            raise PlacementError("bfloat16 was requested but its CUDA execution probe failed")
        return "bfloat16"
    if policy.dtype == "float32":
        return "float32"
    # Pascal-class GPUs have no Tensor Cores. Keep AUTO numerically conservative;
    # explicit float16 remains available when the real execution probe succeeds.
    capability = diagnostics.compute_capability or "0.0"
    try:
        major = int(capability.split(".", 1)[0])
    except ValueError:
        major = 0
    return "float16" if major >= 7 and diagnostics.float16_execution_success else "float32"


def decide_placement(
    policy: RuntimePolicy,
    diagnostics: CudaDiagnostics,
    model_storage_bytes: int,
    *,
    sequential_offload_supported: bool,
    workspace_mib: int = DEFAULT_WORKSPACE_MIB,
) -> PlacementDecision:
    """Choose CPU, full CUDA, or true sequential CPU offload deterministically."""

    policy.validate()
    if model_storage_bytes < 0 or workspace_mib < 0:
        raise PlacementError("model and workspace estimates cannot be negative")
    reserve_bytes = policy.gpu_reserve_mib * MIB
    maximum_bytes = policy.max_vram_mib * MIB if policy.max_vram_mib is not None else None
    budget = effective_gpu_budget_bytes(
        diagnostics.free_vram_bytes_at_startup,
        policy.gpu_reserve_mib,
        policy.max_vram_mib,
    )
    workspace_bytes = workspace_mib * MIB

    def decision(mode: str, dtype: str, reason: str) -> PlacementDecision:
        estimated_model = (
            model_storage_bytes // 2 if dtype in {"float16", "bfloat16"} else model_storage_bytes
        )
        return PlacementDecision(
            execution_mode=mode,
            device="cuda" if mode != "cpu" else "cpu",
            execution_device="cuda" if mode != "cpu" else "cpu",
            offload_mode="sequential-cpu" if mode == "sequential-cpu-offload" else "none",
            dtype=dtype,
            reason=reason,
            configured_gpu_reserve_bytes=reserve_bytes,
            configured_max_vram_bytes=maximum_bytes,
            effective_gpu_budget_bytes=budget,
            estimated_model_bytes=estimated_model,
            estimated_workspace_bytes=workspace_bytes,
            full_cuda_required_bytes=estimated_model + workspace_bytes,
            sequential_offload_supported=sequential_offload_supported,
        )

    if policy.device == "cpu":
        dtype = "float32" if policy.dtype == "auto" else policy.dtype
        if dtype != "float32":
            raise PlacementError("CPU execution currently supports float32 only")
        return decision("cpu", dtype, "CPU was explicitly requested")

    cuda_usable = diagnostics.cuda_available and diagnostics.cuda_kernel_smoke_test_success
    if not cuda_usable:
        reason = diagnostics.cuda_kernel_smoke_test_error or "CUDA execution probe failed"
        if policy.device == "cuda" or policy.offload == "sequential-cpu":
            raise PlacementError(f"CUDA execution is unusable: {reason}")
        return decision("cpu", "float32", f"AUTO selected CPU because {reason}")

    dtype = _cuda_dtype(policy, diagnostics)
    estimated_model = (
        model_storage_bytes // 2 if dtype in {"float16", "bfloat16"} else model_storage_bytes
    )
    required = estimated_model + workspace_bytes
    fits = required <= budget
    if policy.offload == "sequential-cpu":
        if not sequential_offload_supported:
            raise PlacementError("sequential CPU offload is unsupported by this provider")
        return decision(
            "sequential-cpu-offload",
            dtype,
            "sequential CPU offload was explicitly requested",
        )
    if policy.offload == "none":
        if policy.device == "auto" and not fits:
            return decision(
                "cpu",
                "float32",
                "AUTO selected CPU because full CUDA exceeds the effective budget "
                "and offload is disabled",
            )
        return decision(
            "full-cuda",
            dtype,
            "full CUDA was explicitly requested"
            if policy.device == "cuda"
            else "full CUDA fits the effective budget",
        )
    if fits:
        return decision("full-cuda", dtype, "full CUDA fits the effective GPU budget")
    if sequential_offload_supported:
        return decision(
            "sequential-cpu-offload",
            dtype,
            "full CUDA exceeds the effective GPU budget; using CPU-backed sequential execution",
        )
    if policy.device == "auto":
        return decision(
            "cpu",
            "float32",
            "AUTO selected CPU because full CUDA exceeds budget and provider "
            "offload is unsupported",
        )
    raise PlacementError(
        "full CUDA exceeds the effective GPU budget and sequential offload is unsupported"
    )


def torch_dtype(torch: Any, name: str) -> Any:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise PlacementError(f"unsupported Torch dtype: {name}") from exc


def parameter_storage_bytes(model: Any) -> int:
    return int(sum(item.numel() * item.element_size() for item in model.parameters()))


def apply_placement(
    model: Any,
    torch: Any,
    decision: PlacementDecision,
    *,
    preload_module_classes: tuple[str, ...] = (),
    cpu_offload_fn: Callable[..., Any] | None = None,
) -> Any:
    dtype = torch_dtype(torch, decision.dtype)
    model.to(device="cpu", dtype=dtype)
    if decision.execution_mode == "cpu":
        return model
    if decision.execution_mode == "full-cuda":
        return model.to("cuda")
    if decision.execution_mode != "sequential-cpu-offload":
        raise PlacementError(f"unsupported execution mode: {decision.execution_mode}")
    if cpu_offload_fn is None:
        try:
            from accelerate import cpu_offload as cpu_offload_fn
        except ImportError as exc:
            raise PlacementError(
                "sequential CPU offload requires Accelerate in the provider environment"
            ) from exc
    return cpu_offload_fn(
        model,
        execution_device=torch.device("cuda"),
        offload_buffers=True,
        preload_module_classes=list(preload_module_classes) or None,
    )


def restore_model_to_cpu(
    model: Any,
    torch: Any,
    *,
    remove_hooks_fn: Callable[[Any], Any] | None = None,
) -> Any:
    if any(hasattr(module, "_hf_hook") for module in model.modules()):
        if remove_hooks_fn is None:
            from accelerate.hooks import remove_hook_from_submodules as remove_hooks_fn

        remove_hooks_fn(model)
    model.to("cpu")
    gc.collect()
    if bool(torch.cuda.is_available()):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return model


def is_cuda_oom(torch: Any, exc: BaseException) -> bool:
    oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", ())
    return (isinstance(oom_type, type) and isinstance(exc, oom_type)) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def fallback_policy_after_oom(
    policy: RuntimePolicy,
    current_mode: str,
    *,
    sequential_offload_supported: bool,
) -> RuntimePolicy | None:
    """Return the sole bounded AUTO retry policy, or None for explicit modes."""

    if policy.device != "auto":
        return None
    if current_mode == "full-cuda" and sequential_offload_supported:
        return RuntimePolicy(
            device="cuda",
            offload="sequential-cpu",
            gpu_reserve_mib=policy.gpu_reserve_mib,
            max_vram_mib=policy.max_vram_mib,
            dtype=policy.dtype,
        )
    if current_mode in {"full-cuda", "sequential-cpu-offload"}:
        return RuntimePolicy(device="cpu", offload="none", dtype="float32")
    return None


def offload_evidence(model: Any, *, inference_completed: bool) -> dict[str, Any]:
    hooks = 0
    meta_bytes = 0
    cuda_bytes = 0
    devices: dict[str, int] = {}
    for module in model.modules():
        hooks += int(hasattr(module, "_hf_hook"))
    for parameter in model.parameters():
        device = str(parameter.device)
        devices[device] = devices.get(device, 0) + 1
        size = int(parameter.numel() * parameter.element_size())
        if device == "meta":
            meta_bytes += size
        elif device.startswith("cuda"):
            cuda_bytes += size
    return {
        "sequential_offload_hook_count": hooks,
        "offloaded_meta_parameter_bytes": meta_bytes,
        "persistent_cuda_parameter_bytes": cuda_bytes,
        "parameter_device_counts": dict(sorted(devices.items())),
        "sequential_offload_verified": bool(
            inference_completed and hooks > 0 and meta_bytes > 0 and cuda_bytes == 0
        ),
    }


def reset_cuda_peaks(torch: Any) -> None:
    if bool(torch.cuda.is_available()):
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)


def cuda_memory_snapshot(torch: Any) -> dict[str, int | None]:
    if not bool(torch.cuda.is_available()):
        return {
            "cuda_allocated_bytes": None,
            "cuda_reserved_bytes": None,
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
            "cuda_driver_free_bytes": None,
        }
    torch.cuda.synchronize(0)
    free, _total = torch.cuda.mem_get_info(0)
    return {
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated(0)),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved(0)),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "cuda_driver_free_bytes": int(free),
    }


def rss_peak_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024
