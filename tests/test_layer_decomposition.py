from __future__ import annotations

import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from gimp_local_mcp.config import Config
from gimp_local_mcp.gimp.masks import LayerMaskInfo
from gimp_local_mcp.service import GimpService
from gimp_local_mcp.vision.artifacts import (
    complement_mask_png,
    mask_overlap_statistics,
    read_mask_png,
    union_mask_png,
    write_mask_png,
)
from gimp_local_mcp.vision.client import VisionClient
from gimp_local_mcp.vision.protocol import VisionOutOfMemoryError

FIXTURE = Path(__file__).parent / "fixtures" / "fake_vision_worker.py"


def test_soft_mask_complement_union_and_overlap_are_exact(tmp_path: Path) -> None:
    first = write_mask_png(tmp_path / "first.png", 4, 1, bytes([0, 64, 128, 255]))
    second = write_mask_png(tmp_path / "second.png", 4, 1, bytes([0, 128, 128, 0]))
    complement = complement_mask_png(first.path, tmp_path / "complement.png")
    union = union_mask_png([first.path, second.path], tmp_path / "union.png")

    assert read_mask_png(complement.path)[1] == bytes([255, 191, 127, 0])
    expected_union = bytes(
        255 - round((255 - left) * (255 - right) / 255)
        for left, right in zip([0, 64, 128, 255], [0, 128, 128, 0], strict=True)
    )
    assert read_mask_png(union.path)[1] == expected_union
    overlap = mask_overlap_statistics([first.path, second.path])
    assert overlap["overlap_pixel_count"] == 2
    assert overlap["overlap_pixel_ratio"] == 0.5
    assert overlap["overlap_alpha_mass"] == pytest.approx((64 + 128) / 255)


def _decomposition_service(
    *, fail_apply: int | None = None
) -> tuple[GimpService, dict[str, object]]:
    service = GimpService.__new__(GimpService)
    config = Config(
        vision_provider="fake",
        vision_command=(sys.executable, str(FIXTURE)),
        vision_timeout=2,
    )
    service.config = config
    service.vision = VisionClient(config)
    state: dict[str, object] = {
        "next_layer": 100,
        "renames": [],
        "moves": [],
        "applied": [],
        "deleted": [],
        "visibility": [],
        "selected": [],
        "groups": [],
    }
    source = {
        "layer_id": 2,
        "image_id": 1,
        "name": "Original.jpg",
        "width": 8,
        "height": 8,
        "visible": True,
        "opacity": 100.0,
        "mode": 28,
        "position": 0,
        "parent_id": None,
        "is_group": False,
    }
    service._assert_item_belongs_to_image = lambda image_id, layer_id: None  # type: ignore[method-assign]
    service.get_layer_info = lambda image_id, layer_id: dict(source)  # type: ignore[method-assign]
    service.masks = SimpleNamespace(
        get=lambda layer_id: LayerMaskInfo(layer_id, None, False, None, None, None, None, None)
    )
    service._selection_snapshot = lambda image_id: {"active": False}  # type: ignore[method-assign]
    service.get_selected_layers = lambda image_id: [dict(source)]  # type: ignore[method-assign]

    @contextmanager
    def snapshot(image_id: int) -> Iterator[tuple[Path, Path]]:
        with tempfile.TemporaryDirectory(prefix="gimp-mcp-decomposition-test-") as directory:
            root = Path(directory)
            image = write_mask_png(root / "input.png", 8, 8, bytes([255]) * 64)
            yield image.path, root

    service._vision_snapshot = snapshot  # type: ignore[method-assign]

    def create_group(image_id: int, name: str) -> dict[str, object]:
        state["groups"].append(name)  # type: ignore[union-attr]
        return {"layer_id": 900}

    def duplicate(image_id: int, layer_id: int) -> dict[str, object]:
        generated_id = int(state["next_layer"])
        state["next_layer"] = generated_id + 1
        return {"layer_id": generated_id}

    def apply_mask(
        image_id: int,
        layer_id: int,
        path: Path,
        width: int,
        height: int,
        bridge_directory: Path,
    ) -> dict[str, object]:
        applied = state["applied"]
        assert isinstance(applied, list)
        applied.append((layer_id, read_mask_png(path)[1]))
        if fail_apply is not None and len(applied) == fail_apply:
            raise RuntimeError("simulated mask import failure")
        return {"mask_id": layer_id + 1000}

    service.create_layer_group = create_group  # type: ignore[method-assign]
    service.duplicate_layer = duplicate  # type: ignore[method-assign]
    service.rename_layer = lambda layer_id, name: state["renames"].append(  # type: ignore[method-assign,union-attr]
        (layer_id, name)
    )
    service.move_layer = lambda image_id, layer_id, position, parent_id: state[  # type: ignore[method-assign,union-attr]
        "moves"
    ].append((layer_id, position, parent_id))
    service._apply_vision_mask = apply_mask  # type: ignore[method-assign]
    service.set_layer_visibility = lambda layer_id, visible: state["visibility"].append(  # type: ignore[method-assign,union-attr]
        (layer_id, visible)
    )
    service.set_selected_layers = lambda image_id, layer_ids: state["selected"].append(  # type: ignore[method-assign,union-attr]
        list(layer_ids)
    )
    service.delete_layer = lambda image_id, layer_id: state["deleted"].append(  # type: ignore[method-assign,union-attr]
        layer_id
    )
    return service, state


def test_subject_decomposition_creates_named_complementary_layers() -> None:
    service, state = _decomposition_service()
    try:
        result = service.separate_subject_to_layers(1, 2, "red fox")
    finally:
        service.vision.close()

    assert state["groups"] == ["MCP Vision — red fox"]
    assert [name for _layer, name in state["renames"]] == [
        "Subject — red fox",
        "Background",
    ]
    assert result["source_preserved"] is True
    assert result["source_visibility_changed"] is True
    assert state["visibility"] == [(2, False)]
    assert state["selected"] == [[100]]
    subject = state["applied"][0][1]  # type: ignore[index]
    background = state["applied"][1][1]  # type: ignore[index]
    assert all(left + right == 255 for left, right in zip(subject, background, strict=True))


def test_multi_concept_decomposition_reports_overlap_and_remainder() -> None:
    service, state = _decomposition_service()
    try:
        result = service.separate_concepts_to_layers(1, 2, ["red fox", "snowbank"])
    finally:
        service.vision.close()

    assert [name for _layer, name in state["renames"]] == [
        "red fox",
        "snowbank",
        "Remainder Background",
    ]
    assert result["generated_layer_count"] == 3
    assert result["overlap"]["overlap_pixel_count"] > 0
    assert result["overlap_policy"] == "report"
    assert state["selected"] == [[100, 101]]


def test_instance_mode_keeps_provider_instances_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_VISION_INSTANCES", "2")
    service, state = _decomposition_service()
    try:
        result = service.separate_concepts_to_layers(1, 2, ["fox"], include_background=False)
    finally:
        service.vision.close()

    assert [name for _layer, name in state["renames"]] == ["fox 1", "fox 2"]
    assert result["generated_layer_count"] == 2
    assert result["selected_layer_ids"] == [100, 101]


def test_provider_oom_happens_before_gimp_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_VISION_MODE", "oom")
    service, state = _decomposition_service()
    try:
        with pytest.raises(VisionOutOfMemoryError, match="out of memory"):
            service.separate_subject_to_layers(1, 2, "fox")
    finally:
        service.vision.close()
    assert state["groups"] == []
    assert state["renames"] == []


def test_mask_import_failure_rolls_back_generated_layers_and_group() -> None:
    service, state = _decomposition_service(fail_apply=2)
    try:
        with pytest.raises(RuntimeError, match="mask import"):
            service.separate_subject_to_layers(1, 2, "fox")
    finally:
        service.vision.close()
    assert state["deleted"] == [101, 100, 900]
    assert state["visibility"] == []
    assert state["selected"] == [[2]]


def test_decomposition_layer_bound_is_checked_before_provider_work() -> None:
    service, _state = _decomposition_service()
    try:
        with pytest.raises(ValueError, match="at most 24"):
            service.separate_concepts_to_layers(
                1,
                2,
                ["fox", "tree", "snow"],
                max_instances_per_concept=8,
            )
    finally:
        service.vision.close()
