from __future__ import annotations

from unittest.mock import Mock

import pytest

from gimp_local_mcp.config import Config
from gimp_local_mcp.errors import ScriptFuError
from gimp_local_mcp.gimp.serializer import SchemeVector
from gimp_local_mcp.models import LayerInfo, LayerNode
from gimp_local_mcp.service import GimpService, _strict_object_ids


def test_strict_object_ids_accepts_gimp_array_shapes() -> None:
    assert _strict_object_ids([1, 2], "layers") == [1, 2]
    assert _strict_object_ids([[3, 4]], "layers") == [3, 4]
    assert _strict_object_ids([2, [5, 6]], "layers") == [5, 6]


@pytest.mark.parametrize("value", ["bad", ["bad"], [1, -1], [True]])
def test_strict_object_ids_rejects_malformed_values(value: object) -> None:
    with pytest.raises(RuntimeError, match="malformed layers"):
        _strict_object_ids(value, "layers")


def _fake_state_service(children: list[int] | None = None) -> GimpService:
    service = GimpService.__new__(GimpService)
    child_ids = [11] if children is None else children

    def evaluate(expression: str) -> object:
        if expression == "(gimp-image-get-layers 1)":
            return [10, 20]
        if expression == "(gimp-item-get-children 10)":
            return child_ids
        if expression == "(gimp-item-get-children 20)":
            return []
        if expression == "(gimp-image-get-selected-layers 1)":
            return [11, 20]
        if expression == "(gimp-item-get-image 10)" or expression == "(gimp-item-get-image 11)":
            return 1
        if expression == "(gimp-item-get-image 20)":
            return 1
        if expression == "(gimp-image-set-selected-layers 1 #(11 20))":
            return []
        if expression.startswith("(list "):
            if " 10)" in expression:
                parent = 10 if child_ids == [10] else -1
                return ["Group", 16, 16, True, 100.0, 28, parent, True, 1]
            if " 11)" in expression:
                return ["Child", 16, 16, True, 100.0, 28, 10, False, 1]
            if " 20)" in expression:
                return ["Other", 16, 16, True, 100.0, 28, -1, False, 1]
        if expression in {
            "(gimp-image-get-item-position 1 10)",
            "(gimp-image-get-item-position 10)",
        }:
            return 0
        if expression in {
            "(gimp-image-get-item-position 1 11)",
            "(gimp-image-get-item-position 11)",
        }:
            return 0
        if expression in {
            "(gimp-image-get-item-position 1 20)",
            "(gimp-image-get-item-position 20)",
        }:
            return 1
        raise AssertionError(expression)

    service.evaluate = Mock(side_effect=evaluate)  # type: ignore[method-assign]
    service._fake_evaluate = evaluate  # type: ignore[attr-defined]
    return service


def test_layer_tree_is_recursive_and_preserves_parent_relationships() -> None:
    tree = _fake_state_service().get_layer_tree(1)
    assert tree[0]["layer_id"] == 10
    assert tree[0]["is_group"] is True
    assert tree[0]["children"][0]["layer_id"] == 11
    assert tree[0]["children"][0]["parent_id"] == 10
    assert tree[1]["layer_id"] == 20
    assert tree[1]["children"] == []


def test_layer_tree_rejects_cycles_and_item_limits() -> None:
    cycle = _fake_state_service()
    root = {
        "layer_id": 10,
        "image_id": 1,
        "name": "Group",
        "width": 16,
        "height": 16,
        "visible": True,
        "opacity": 100.0,
        "mode": 28,
        "position": 0,
        "parent_id": None,
        "is_group": True,
    }
    child = {**root, "parent_id": 10}
    cycle.list_layers = Mock(return_value=[root])  # type: ignore[method-assign]
    cycle.get_layer_info = Mock(side_effect=[child])  # type: ignore[method-assign]
    cycle.evaluate.side_effect = lambda expression: (
        [10] if expression == "(gimp-item-get-children 10)" else cycle._fake_evaluate(expression)
    )  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="cyclic"):
        cycle.get_layer_tree(1)
    with pytest.raises(RuntimeError, match="item-count"):
        _fake_state_service().get_layer_tree(1, max_items=1)


def test_selected_layers_support_multiple_layers_and_read_back() -> None:
    service = _fake_state_service()
    selected = service.get_selected_layers(1)
    assert [item["layer_id"] for item in selected] == [11, 20]
    result = service.set_selected_layers(1, [11, 20])
    assert result["selection_model"] == "multi-layer"
    assert result["selected_layer_ids"] == [11, 20]


def test_selection_rejects_empty_and_duplicate_ids() -> None:
    service = _fake_state_service()
    with pytest.raises(ValueError, match="non-empty"):
        service.set_selected_layers(1, [])
    with pytest.raises(ValueError, match="duplicates"):
        service.set_selected_layers(1, [11, 11])


def test_current_context_reports_single_image_fallback_and_selection() -> None:
    service = GimpService.__new__(GimpService)
    service.config = Config()
    image = {
        "image_id": 1,
        "name": "Untitled",
        "width": 8,
        "height": 8,
        "base_type": 0,
        "file": None,
        "dirty": False,
    }
    selected = [{"layer_id": 11, "image_id": 1, "name": "Layer"}]
    service.list_open_images = Mock(return_value=[image])  # type: ignore[method-assign]
    service.get_selected_layers = Mock(return_value=selected)  # type: ignore[method-assign]

    def evaluate(expression: str) -> object:
        if expression == "(gimp-version)":
            return "3.2.0"
        if expression == "(gimp-default-display)":
            raise ScriptFuError("unbound variable")
        raise AssertionError(expression)

    service.evaluate = Mock(side_effect=evaluate)  # type: ignore[method-assign]
    context = service.get_current_context()
    assert context["current_image"] == image
    assert context["current_image_source"] == "single-open-image"
    assert context["default_display_id"] is None
    assert context["selected_layer_ids"] == [11]


def test_layer_node_is_json_compatible() -> None:
    node = LayerNode(
        LayerInfo(1, 2, "Group", 10, 10, True, 100, 28, 0, None, True),
        (LayerNode(LayerInfo(3, 2, "Child", 10, 10, True, 100, 28, 0, 1, False)),),
    )
    assert node.as_dict()["children"][0]["layer_id"] == 3


def test_scheme_vector_is_structured_not_source() -> None:
    assert SchemeVector((4, 5)).values == (4, 5)
