from __future__ import annotations

from types import SimpleNamespace

from gimp_local_mcp import cli


def test_doctor_distinguishes_driver_torch_cuda_and_model_readiness(monkeypatch, capsys) -> None:
    class FakeService:
        def __init__(self, config) -> None:
            pass

        def vision_status(self):
            return {
                "provider": "mock",
                "available": True,
                "reason": None,
                "torch_version": "2.test",
                "torch_cuda_version": "12.6",
                "cuda_available": True,
                "device": "cuda",
                "gpu_name": "Mock GPU",
                "compute_capability": "8.6",
                "checkpoint_available": True,
                "model_load_success": True,
                "test_inference_success": True,
                "text_segmentation": True,
            }

        def status(self):
            return {"gimp_version": "3.2.0", "open_image_count": 1}

        def close(self) -> None:
            pass

    monkeypatch.setattr("gimp_local_mcp.service.GimpService", FakeService)
    monkeypatch.setattr(cli.shutil, "which", lambda command: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Mock GPU, 999.1, 4096, 512, 3584, 8.6\n",
        ),
    )

    assert cli._doctor() == 0
    output = capsys.readouterr().out
    assert "NVIDIA driver: detected" in output
    assert "torch CUDA=12.6" in output
    assert "CUDA available=True" in output
    assert "model load=True" in output
    assert "test inference=True" in output
