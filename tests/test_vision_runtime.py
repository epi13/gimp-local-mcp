from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from tools.vision.clipseg_worker import (  # noqa: E402
    WorkerUnavailable,
    clipseg_mask_settings,
)
from tools.vision.runtime import (  # noqa: E402
    MIB,
    CudaDiagnostics,
    PlacementError,
    RuntimePolicy,
    apply_placement,
    decide_placement,
    effective_gpu_budget_bytes,
    fallback_policy_after_oom,
)


def test_clipseg_mask_settings_preserve_defaults() -> None:
    assert clipseg_mask_settings({}) == (0.2, 2.0)


def test_clipseg_mask_settings_accept_bounded_overrides() -> None:
    assert clipseg_mask_settings(
        {
            "GIMP_MCP_CLIPSEG_MASK_THRESHOLD": "0.30",
            "GIMP_MCP_CLIPSEG_MASK_SLOPE": "3",
        }
    ) == (0.3, 3.0)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("GIMP_MCP_CLIPSEG_MASK_THRESHOLD", "0"),
        ("GIMP_MCP_CLIPSEG_MASK_THRESHOLD", "1"),
        ("GIMP_MCP_CLIPSEG_MASK_SLOPE", "0.1"),
        ("GIMP_MCP_CLIPSEG_MASK_SLOPE", "9"),
        ("GIMP_MCP_CLIPSEG_MASK_SLOPE", str(math.inf)),
        ("GIMP_MCP_CLIPSEG_MASK_THRESHOLD", "not-a-number"),
    ],
)
def test_clipseg_mask_settings_reject_invalid_values(name: str, value: str) -> None:
    with pytest.raises(WorkerUnavailable, match=name):
        clipseg_mask_settings({name: value})


def diagnostics(*, usable: bool = True, free_mib: int = 1800) -> CudaDiagnostics:
    return CudaDiagnostics(
        torch_version="2.test",
        torch_cuda_version="12.6",
        cuda_available=usable,
        cuda_kernel_smoke_test_success=usable,
        cuda_kernel_smoke_test_error=None if usable else "no kernel image",
        gpu_name="Fake GPU",
        compute_capability="6.1",
        torch_supported_architectures=("sm_60",),
        total_vram_bytes=2048 * MIB,
        free_vram_bytes_at_startup=free_mib * MIB,
        float16_execution_success=usable,
        bfloat16_execution_success=False,
    )


def test_effective_budget_applies_cap_then_reserve() -> None:
    assert effective_gpu_budget_bytes(1800 * MIB, 256, None) == 1544 * MIB
    assert effective_gpu_budget_bytes(1800 * MIB, 256, 1024) == 768 * MIB
    assert effective_gpu_budget_bytes(128 * MIB, 256, None) == 0


def test_auto_selects_full_cuda_when_model_and_workspace_fit() -> None:
    decision = decide_placement(
        RuntimePolicy(), diagnostics(), 500 * MIB, sequential_offload_supported=True
    )
    assert decision.execution_mode == "full-cuda"
    assert decision.dtype == "float32"


def test_auto_selects_true_sequential_offload_when_full_model_does_not_fit() -> None:
    decision = decide_placement(
        RuntimePolicy(),
        diagnostics(free_mib=1000),
        900 * MIB,
        sequential_offload_supported=True,
    )
    assert decision.execution_mode == "sequential-cpu-offload"
    assert decision.execution_device == "cuda"
    assert decision.offload_mode == "sequential-cpu"


def test_auto_falls_back_to_cpu_when_cuda_kernel_is_unusable() -> None:
    decision = decide_placement(
        RuntimePolicy(), diagnostics(usable=False), 1, sequential_offload_supported=True
    )
    assert decision.execution_mode == "cpu"
    assert "no kernel image" in decision.reason


def test_explicit_cuda_does_not_silently_fall_back() -> None:
    with pytest.raises(PlacementError, match="unusable"):
        decide_placement(
            RuntimePolicy(device="cuda", offload="none"),
            diagnostics(usable=False),
            1,
            sequential_offload_supported=True,
        )


def test_explicit_modes_and_unsupported_offload() -> None:
    cpu = decide_placement(
        RuntimePolicy(device="cpu", offload="none"),
        diagnostics(),
        999 * MIB,
        sequential_offload_supported=False,
    )
    assert cpu.execution_mode == "cpu"
    with pytest.raises(PlacementError, match="unsupported"):
        decide_placement(
            RuntimePolicy(device="cuda", offload="sequential-cpu"),
            diagnostics(),
            999 * MIB,
            sequential_offload_supported=False,
        )


def test_oom_fallback_is_typed_bounded_and_auto_only() -> None:
    policy = RuntimePolicy()
    first = fallback_policy_after_oom(policy, "full-cuda", sequential_offload_supported=True)
    assert first is not None and first.offload == "sequential-cpu"
    second = fallback_policy_after_oom(
        policy, "sequential-cpu-offload", sequential_offload_supported=True
    )
    assert second is not None and second.device == "cpu"
    explicit = replace(policy, device="cuda", offload="none")
    assert (
        fallback_policy_after_oom(explicit, "full-cuda", sequential_offload_supported=True) is None
    )


def test_apply_placement_uses_accelerate_cpu_offload_contract() -> None:
    calls: dict[str, object] = {}

    class Device:
        def __init__(self, value: str) -> None:
            self.value = value

    class Torch:
        float32 = "float32"
        float16 = "float16"
        bfloat16 = "bfloat16"
        device = Device

    class Model:
        def __init__(self) -> None:
            self.moves: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def to(self, *args: object, **kwargs: object) -> Model:
            self.moves.append((args, kwargs))
            return self

    def offload(model: Model, **kwargs: object) -> Model:
        calls.update(kwargs)
        return model

    decision = decide_placement(
        RuntimePolicy(device="cuda", offload="sequential-cpu"),
        diagnostics(),
        900 * MIB,
        sequential_offload_supported=True,
    )
    model = Model()
    assert apply_placement(model, Torch(), decision, cpu_offload_fn=offload) is model
    assert calls["offload_buffers"] is True
    assert isinstance(calls["execution_device"], Device)
    assert calls["preload_module_classes"] is None
