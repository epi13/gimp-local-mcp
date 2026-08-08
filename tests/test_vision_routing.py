from __future__ import annotations

from types import SimpleNamespace

import pytest

from gimp_local_mcp.service import GimpService
from gimp_local_mcp.vision.models import VisionCapabilities


def test_auto_prefers_semantic_provider_when_capable() -> None:
    service = GimpService.__new__(GimpService)
    service.vision = SimpleNamespace(
        capabilities=lambda: VisionCapabilities("fake", True, text_segmentation=True)
    )
    service.isolate_subject_vision = lambda *args, **kwargs: {
        "strategy": "vision",
        "prompt": args[2],
    }
    service._isolate_subject_heuristic = lambda *args: {"strategy": "heuristic"}
    assert service.isolate_subject(1, 2, strategy="auto", prompt="red fox") == {
        "strategy": "vision",
        "prompt": "red fox",
    }


def test_auto_falls_back_when_provider_unavailable() -> None:
    service = GimpService.__new__(GimpService)
    service.vision = SimpleNamespace(
        capabilities=lambda: VisionCapabilities("fake", False, reason="not installed")
    )
    service.isolate_subject_vision = lambda *args, **kwargs: {"strategy": "vision"}
    service._isolate_subject_heuristic = lambda *args: {"strategy": args[2]}
    assert service.isolate_subject(1, 2, strategy="auto") == {"strategy": "auto"}


@pytest.mark.parametrize("strategy", ["ml", "snow", "", "../worker"])
def test_vision_strategy_vocabulary_is_bounded(strategy: str) -> None:
    service = GimpService.__new__(GimpService)
    with pytest.raises(ValueError, match="strategy"):
        service.isolate_subject(1, 2, strategy=strategy)


def test_mask_artifact_path_cannot_escape_provider_directory(tmp_path) -> None:
    service = GimpService.__new__(GimpService)
    candidate = SimpleNamespace(mask=SimpleNamespace(path=tmp_path.parent / "outside.png"))
    result = SimpleNamespace(candidates=[candidate])
    with pytest.raises(RuntimeError, match="escaped"):
        service._validate_vision_result(result, tmp_path, 8, 8)
