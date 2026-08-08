from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.vision import sam3_worker  # noqa: E402

WORKER = REPOSITORY_ROOT / "tools" / "vision" / "sam3_worker.py"


def _request(message: dict[str, object]) -> dict[str, object]:
    environment = os.environ.copy()
    environment.pop("GIMP_MCP_SAM3_CHECKPOINT", None)
    result = subprocess.run(
        [sys.executable, str(WORKER)],
        input=json.dumps(message) + "\n",
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
        env=environment,
    )
    return json.loads(result.stdout)


def test_sam3_reports_missing_official_setup_without_downloading() -> None:
    response = _request({"protocol_version": "0.1", "type": "capabilities", "request_id": "status"})
    capabilities = response["capabilities"]
    assert isinstance(capabilities, dict)
    assert capabilities["provider"] == "sam3"
    assert capabilities["available"] is False
    assert capabilities["checkpoint_available"] is False
    assert capabilities["text_segmentation"] is False
    assert capabilities["visual_prompts"] is False
    assert "not installed" in str(capabilities["reason"])


def test_sam3_segment_fails_closed_when_provider_is_unavailable() -> None:
    response = _request(
        {
            "protocol_version": "0.1",
            "type": "segment",
            "request_id": "segment",
            "request": {},
        }
    )
    assert response["type"] == "error"
    assert response["code"] == "unavailable"


def test_sam3_official_builder_is_forced_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "sam3.1.pt"
    checkpoint.write_bytes(b"fake checkpoint")
    calls: dict[str, object] = {}

    class Parameter:
        device = "cpu"

        @staticmethod
        def numel() -> int:
            return 4

        @staticmethod
        def element_size() -> int:
            return 4

    class Model:
        def parameters(self) -> list[Parameter]:
            return [Parameter()]

        def modules(self) -> list[Model]:
            return [self]

        def to(self, *args: object, **kwargs: object) -> Model:
            return self

        def eval(self) -> Model:
            return self

    class Processor:
        def __init__(self, model: Model, **kwargs: object) -> None:
            self.model = model

        @staticmethod
        def set_image(image: object) -> dict[str, object]:
            return {"backbone_out": object()}

        @staticmethod
        def set_text_prompt(prompt: str, state: dict[str, object]) -> dict[str, object]:
            state.update(masks=[], boxes=[], scores=[])
            return state

        @staticmethod
        def add_geometric_prompt(
            box: list[float], label: bool, state: dict[str, object]
        ) -> dict[str, object]:
            state.update(masks=[], boxes=[], scores=[])
            return state

    def builder(**kwargs: object) -> Model:
        calls.update(kwargs)
        return Model()

    model_builder = types.ModuleType("sam3.model_builder")
    model_builder.build_sam3_image_model = builder  # type: ignore[attr-defined]
    processor_module = types.ModuleType("sam3.model.sam3_image_processor")
    processor_module.Sam3Processor = Processor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sam3", types.ModuleType("sam3"))
    monkeypatch.setitem(sys.modules, "sam3.model", types.ModuleType("sam3.model"))
    monkeypatch.setitem(sys.modules, "sam3.model_builder", model_builder)
    monkeypatch.setitem(sys.modules, "sam3.model.sam3_image_processor", processor_module)
    monkeypatch.setattr(sam3_worker, "CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("GIMP_MCP_VISION_DEVICE", "cpu")
    monkeypatch.setenv("GIMP_MCP_VISION_OFFLOAD", "none")

    runtime = sam3_worker.Sam3Runtime()
    runtime._diagnose_torch()
    assert runtime.torch is not None
    monkeypatch.setattr(sam3_worker.importlib.util, "find_spec", lambda name: object())
    runtime.ensure_loaded()
    assert calls["load_from_HF"] is False
    assert calls["checkpoint_path"] == str(checkpoint)
    assert calls["device"] == "cpu"
    assert runtime.text_tested is True
    assert runtime.box_tested is True
