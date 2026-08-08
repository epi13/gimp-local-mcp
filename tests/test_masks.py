from __future__ import annotations

import math

import pytest

from gimp_local_mcp.gimp.masks import LayerMaskGateway
from gimp_local_mcp.gimp.metrics import mask_metrics
from gimp_local_mcp.gimp.serializer import SchemeSymbol, scheme_call


def test_mask_calls_use_structured_enum_serialization() -> None:
    assert scheme_call("gimp-layer-create-mask", [12, SchemeSymbol("ADD-MASK-SELECTION")]) == (
        "(gimp-layer-create-mask 12 ADD-MASK-SELECTION)"
    )


def test_mask_gateway_creates_attaches_and_reads_state() -> None:
    state = {"mask": -1, "enabled": True, "visible": False, "editable": True}

    def evaluate(expression: str) -> object:
        if expression == "(gimp-layer-get-mask 12)":
            return state["mask"]
        if expression == "(gimp-layer-create-mask 12 ADD-MASK-SELECTION)":
            return 42
        if expression == "(gimp-layer-add-mask 12 42)":
            state["mask"] = 42
            return []
        if expression == "(gimp-layer-get-apply-mask 12)":
            return state["enabled"]
        if expression == "(gimp-layer-get-show-mask 12)":
            return state["visible"]
        if expression == "(gimp-layer-get-edit-mask 12)":
            return state["editable"]
        if expression == "(gimp-drawable-get-width 42)":
            return 16
        if expression == "(gimp-drawable-get-height 42)":
            return 8
        if expression == "(gimp-layer-set-apply-mask 12 #f)":
            state["enabled"] = False
            return []
        raise AssertionError(expression)

    gateway = LayerMaskGateway(evaluate)
    assert gateway.create_and_attach(12).as_dict() == {
        "layer_id": 12,
        "mask_id": 42,
        "has_mask": True,
        "enabled": True,
        "visible": False,
        "editable": True,
        "width": 16,
        "height": 8,
    }
    assert gateway.set_enabled(12, False).enabled is False
    with pytest.raises(ValueError, match="already has a mask"):
        gateway.create(12)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -1, 256, True])
def test_mask_pixel_validation_rejects_malformed_values(value: object) -> None:
    with pytest.raises(RuntimeError, match="mask pixel"):
        LayerMaskGateway.validate_alpha(value)


def test_mask_metrics_reports_partial_alpha_and_border_rejection() -> None:
    samples = [
        (0, 0, 0),
        (1, 0, 32),
        (2, 0, 255),
        (0, 1, 0),
        (1, 1, 128),
        (2, 1, 255),
        (0, 2, 0),
        (1, 2, 255),
        (2, 2, 255),
    ]
    result = mask_metrics(
        samples, 3, 3, foreground_reference=[0, 255, 255, 0, 255, 255, 0, 255, 255]
    )
    assert result["border_transparency_ratio"] == pytest.approx(3 / 8)
    assert result["partial_alpha_ratio"] == pytest.approx(2 / 9)
    assert result["edge_transition_sample_ratio"] > 0
    assert result["foreground_retention_proxy"] == pytest.approx(1.0)
    assert result["retained_bounding_box"] == {
        "x": 1,
        "y": 0,
        "width": 2,
        "height": 3,
        "sampled": True,
    }


@pytest.mark.parametrize("samples", [[(0, 0, 0)], [(0, 0, 255)]])
def test_mask_metrics_flags_pathological_masks(samples: list[tuple[int, int, int]]) -> None:
    result = mask_metrics(samples, 1, 1)
    assert result["all_transparent"] is (samples[0][2] == 0)
    assert result["all_opaque"] is (samples[0][2] == 255)


def test_mask_metrics_rejects_empty_or_misaligned_reference() -> None:
    assert mask_metrics([], 4, 4)["empty_sample"] is True
    with pytest.raises(ValueError, match="align"):
        mask_metrics([(0, 0, 255)], 1, 1, foreground_reference=[])
