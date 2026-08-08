from __future__ import annotations

import math

import pytest

from gimp_local_mcp.gimp.filters import DrawableFilterGateway, DrawableFilterSpec
from gimp_local_mcp.service import GimpService


def test_filter_spec_serializes_structured_gegl_parameters() -> None:
    spec = DrawableFilterSpec.create(
        12,
        "gegl:gaussian-blur",
        'Blur "safe"',
        opacity=40,
        parameters={"std-dev-x": 2.0, "std-dev-y": 3.0},
    )
    assert spec.expression() == (
        '(gimp-drawable-append-new-filter 12 "gegl:gaussian-blur" '
        '"Blur \\"safe\\"" LAYER-MODE-REPLACE 0.40000000000000002 '
        "#:std-dev-x 2 #:std-dev-y 3)"
    )


@pytest.mark.parametrize(
    "operation",
    ['(display "bad")', "gegl:bad operation", "gegl:\\n"],
)
def test_filter_operation_cannot_cross_scheme_boundary(operation: str) -> None:
    with pytest.raises(ValueError):
        DrawableFilterSpec.create(1, operation, "safe")


@pytest.mark.parametrize("opacity", [-1, 101, math.inf, math.nan])
def test_filter_opacity_is_bounded(opacity: float) -> None:
    with pytest.raises(ValueError):
        DrawableFilterSpec.create(1, "gegl:gaussian-blur", "safe", opacity=opacity)


def test_filter_gateway_returns_only_gimp_reported_identity_and_state() -> None:
    def evaluate(expression: str) -> object:
        if expression.startswith("(gimp-drawable-append-new-filter"):
            return 27
        if expression == "(gimp-drawable-filter-get-name 27)":
            return "Blur"
        if expression == "(gimp-drawable-filter-get-operation-name 27)":
            return "gegl:gaussian-blur"
        if expression == "(gimp-drawable-filter-get-opacity 27)":
            return 0.4
        if expression == "(gimp-drawable-filter-get-blend-mode 27)":
            return 62
        if expression == "(gimp-drawable-filter-get-visible 27)":
            return True
        if expression == "(gimp-drawable-get-filters 12)":
            return [27]
        raise AssertionError(expression)

    gateway = DrawableFilterGateway(evaluate)
    spec = DrawableFilterSpec.create(12, "gegl:gaussian-blur", "Blur", parameters={"std-dev-x": 2})
    result = gateway.append(spec).as_dict()
    assert result == {
        "filter_id": 27,
        "drawable_id": 12,
        "name": "Blur",
        "operation": "gegl:gaussian-blur",
        "opacity": 0.4,
        "blend_mode": 62,
        "visible": True,
        "non_destructive": True,
    }
    assert gateway.list(12)[0].filter_id == 27


def test_filter_gateway_rejects_malformed_filter_result() -> None:
    gateway = DrawableFilterGateway(lambda expression: "not-an-id")
    spec = DrawableFilterSpec.create(12, "gegl:gaussian-blur", "Blur")
    with pytest.raises(RuntimeError, match="invalid drawable filter ID"):
        gateway.append(spec)


def test_filter_adjustment_ranges_reject_non_finite_values() -> None:
    with pytest.raises(ValueError):
        GimpService._finite_range(math.nan, -1, 1, "brightness")
    with pytest.raises(ValueError):
        GimpService._finite_range(2, -1, 1, "contrast")
