from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

from gimp_local_mcp.config import Config
from gimp_local_mcp.vision.artifacts import MaskArtifactError, read_mask_png, write_mask_png
from gimp_local_mcp.vision.client import VisionClient
from gimp_local_mcp.vision.models import (
    BoundingBox,
    MaskArtifact,
    SegmentationRequest,
    VisionCapabilities,
)
from gimp_local_mcp.vision.protocol import (
    VisionOutOfMemoryError,
    VisionProtocolError,
    VisionUnavailableError,
    VisionWorkerError,
    decode_message,
    encode_message,
)

FIXTURE = Path(__file__).parent / "fixtures" / "fake_vision_worker.py"


def fake_config(timeout: float = 1.0) -> Config:
    return Config(
        vision_provider="fake",
        vision_command=(sys.executable, str(FIXTURE)),
        vision_timeout=timeout,
    )


def test_mask_png_round_trip_preserves_partial_alpha(tmp_path: Path) -> None:
    path = tmp_path / "mask.png"
    artifact = write_mask_png(path, 4, 2, bytes([0, 64, 128, 255, 255, 192, 1, 0]))
    checked, pixels = read_mask_png(path)
    assert artifact.width == checked.width == 4
    assert checked.soft_alpha is True
    assert pixels == bytes([0, 64, 128, 255, 255, 192, 1, 0])


def test_mask_artifact_rejects_non_png_and_bad_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "bad.png"
    path.write_bytes(b"not png")
    with pytest.raises(MaskArtifactError):
        read_mask_png(path)
    with pytest.raises(ValueError, match="absolute"):
        MaskArtifact(Path("relative.png"), 2, 2)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_vision_models_reject_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        BoundingBox(value, 0, 1, 1)
    with pytest.raises(ValueError, match="finite"):
        SegmentationRequest(Path("/tmp/input.png"), minimum_score=value)


def test_request_and_response_json_are_bounded_and_typed(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    write_mask_png(input_path, 8, 8, bytes([255]) * 64)
    request = SegmentationRequest(input_path, prompt="red fox", output_directory=tmp_path)
    message = {"protocol_version": "0.1", "type": "segment", "request": request.as_dict()}
    assert decode_message(encode_message(message))["type"] == "segment"
    with pytest.raises(VisionProtocolError):
        decode_message(b"{not-json}")
    with pytest.raises(VisionProtocolError):
        encode_message({"protocol_version": "0.0"})


def test_fake_provider_capabilities_and_soft_mask(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    write_mask_png(input_path, 8, 8, bytes([255]) * 64)
    client = VisionClient(fake_config(), command=(sys.executable, str(FIXTURE)))
    try:
        capabilities = client.capabilities()
        assert capabilities.available is True
        result = client.segment(
            SegmentationRequest(input_path, prompt="red fox", output_directory=tmp_path)
        )
        assert result.provider == "fake"
        assert result.candidates[0].score == 0.91
        assert result.candidates[0].mask.path.is_file()
        assert read_mask_png(result.candidates[0].mask.path)[0].soft_alpha is True
    finally:
        client.close()


def test_provider_timeout_is_typed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    input_path = tmp_path / "input.png"
    write_mask_png(input_path, 8, 8, bytes([255]) * 64)
    monkeypatch.setenv("FAKE_VISION_MODE", "sleep")
    client = VisionClient(fake_config(timeout=0.05), command=(sys.executable, str(FIXTURE)))
    try:
        with pytest.raises(VisionWorkerError, match="timed out"):
            client.segment(SegmentationRequest(input_path, prompt="fox", output_directory=tmp_path))
    finally:
        client.close()


def test_provider_oom_and_unavailable_responses_are_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "input.png"
    write_mask_png(input_path, 8, 8, bytes([255]) * 64)
    request = SegmentationRequest(input_path, prompt="fox", output_directory=tmp_path)
    for mode, error_type in (
        ("oom", VisionOutOfMemoryError),
        ("unavailable", VisionUnavailableError),
    ):
        monkeypatch.setenv("FAKE_VISION_MODE", mode)
        client = VisionClient(fake_config(), command=(sys.executable, str(FIXTURE)))
        try:
            with pytest.raises(error_type):
                client.segment(request)
        finally:
            client.close()


def test_fake_capabilities_report_cpu_and_readiness() -> None:
    client = VisionClient(fake_config(), command=(sys.executable, str(FIXTURE)))
    try:
        capabilities = client.capabilities()
    finally:
        client.close()
    assert capabilities.device == "cpu"
    assert capabilities.cuda_available is False
    assert capabilities.checkpoint_available is True
    assert capabilities.model_load_success is True
    assert capabilities.test_inference_success is True


def test_gpu_capability_metadata_is_typed() -> None:
    capabilities = VisionCapabilities.from_dict(
        {
            "provider": "mock-gpu",
            "available": True,
            "text_segmentation": True,
            "cuda_available": True,
            "device": "cuda",
            "gpu_name": "Test GPU",
            "compute_capability": "8.6",
            "total_vram_bytes": 4_000_000_000,
            "available_vram_bytes": 3_000_000_000,
            "model_load_success": True,
            "test_inference_success": True,
        }
    )
    assert capabilities.device == "cuda"
    assert capabilities.cuda_available is True
    assert capabilities.total_vram_bytes == 4_000_000_000
    with pytest.raises(ValueError, match="total_vram"):
        VisionCapabilities.from_dict(
            {"provider": "bad", "available": False, "total_vram_bytes": -1}
        )


def test_extended_placement_capabilities_are_optional_and_bounded() -> None:
    legacy = VisionCapabilities.from_dict({"provider": "legacy", "available": True})
    assert legacy.execution_mode is None
    capabilities = VisionCapabilities.from_dict(
        {
            "provider": "offload",
            "available": True,
            "execution_device": "cuda",
            "execution_mode": "sequential-cpu-offload",
            "offload_mode": "sequential-cpu",
            "torch_supported_architectures": ["sm_60", "sm_70"],
            "cuda_kernel_smoke_test_success": True,
            "sequential_offload_verified": True,
            "offloaded_meta_parameter_bytes": 1234,
            "persistent_cuda_parameter_bytes": 0,
        }
    )
    assert capabilities.torch_supported_architectures == ("sm_60", "sm_70")
    assert capabilities.sequential_offload_verified is True
    assert capabilities.as_dict()["offload_mode"] == "sequential-cpu"
    assert VisionCapabilities.from_dict(legacy.as_dict()) == legacy


def test_malformed_worker_response_becomes_unavailable_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_VISION_MODE", "malformed")
    client = VisionClient(fake_config(), command=(sys.executable, str(FIXTURE)))
    try:
        status = client.capabilities()
        assert status.available is False
        assert "expected capabilities_response" in (status.reason or "")
    finally:
        client.close()


def test_provider_does_not_use_shell_or_accept_mcp_command(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    write_mask_png(input_path, 8, 8, bytes([255]) * 64)
    config = fake_config()
    assert config.vision_command is not None
    assert ";" not in config.vision_command[0]
    request = SegmentationRequest(input_path, prompt="fox", output_directory=tmp_path)
    client = VisionClient(config, command=config.vision_command)
    try:
        assert client.segment(request).provider == "fake"
    finally:
        client.close()
