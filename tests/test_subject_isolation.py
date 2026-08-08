from __future__ import annotations

import math

import pytest

from gimp_local_mcp.service import GimpService


@pytest.mark.parametrize("strategy", ["bad", "snow", "ml"])
def test_subject_isolation_strategy_is_bounded_without_gimp(strategy: str) -> None:
    service = GimpService.__new__(GimpService)
    service._assert_item_belongs_to_image = lambda *_: None  # type: ignore[method-assign]
    service.get_layer_info = lambda *_: {  # type: ignore[method-assign]
        "layer_id": 2,
        "image_id": 1,
        "width": 8,
        "height": 8,
        "visible": True,
    }
    with pytest.raises(ValueError, match="strategy"):
        service.isolate_subject(1, 2, strategy=strategy)


@pytest.mark.parametrize("value", [0, 256, -1, True, math.nan, math.inf])
def test_subject_isolation_thresholds_reject_invalid_values(value: object) -> None:
    service = GimpService.__new__(GimpService)
    service._assert_item_belongs_to_image = lambda *_: None  # type: ignore[method-assign]
    service.get_layer_info = lambda *_: {  # type: ignore[method-assign]
        "layer_id": 2,
        "image_id": 1,
        "width": 8,
        "height": 8,
        "visible": True,
    }
    with pytest.raises(ValueError, match="threshold"):
        service.isolate_subject(1, 2, background_threshold=value)  # type: ignore[arg-type]


def test_subject_isolation_rejects_refinement_above_baseline() -> None:
    service = GimpService.__new__(GimpService)
    service._assert_item_belongs_to_image = lambda *_: None  # type: ignore[method-assign]
    service.get_layer_info = lambda *_: {  # type: ignore[method-assign]
        "layer_id": 2,
        "image_id": 1,
        "width": 8,
        "height": 8,
        "visible": True,
    }
    with pytest.raises(ValueError, match="cannot exceed"):
        service.isolate_subject(1, 2, background_threshold=24, refinement_threshold=48)
