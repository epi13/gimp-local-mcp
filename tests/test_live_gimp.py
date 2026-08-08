from __future__ import annotations

import pytest

from gimp_local_mcp.config import Config
from gimp_local_mcp.errors import GimpConnectionError
from gimp_local_mcp.gimp.metrics import mask_metrics
from gimp_local_mcp.gimp.scheme import parse_scheme, unwrap
from gimp_local_mcp.gimp.transport import ScriptFuClient
from gimp_local_mcp.service import GimpService


def _live_service() -> GimpService:
    config = Config.from_env()
    probe = ScriptFuClient(config)
    try:
        try:
            probe.connect()
        except GimpConnectionError:
            pytest.skip("GIMP Script-Fu server is not available")
        version = unwrap(parse_scheme(probe.execute("(gimp-version)")))
        if isinstance(version, list) and len(version) == 1:
            version = version[0]
        if not isinstance(version, str) or not version.startswith("3."):
            pytest.skip(f"GIMP 3 is required for live state tests; found {version!r}")
    finally:
        probe.close()
    return GimpService(config)


def _flatten_tree(nodes: list[dict[str, object]]) -> list[dict[str, object]]:
    flattened: list[dict[str, object]] = []
    for node in nodes:
        flattened.append(node)
        children = node.get("children")
        if isinstance(children, list):
            flattened.extend(_flatten_tree(children))
    return flattened


@pytest.mark.integration
def test_live_gimp_version() -> None:
    service = _live_service()
    try:
        assert service.status()["gimp_version"].startswith("3.")
    finally:
        service.close()


@pytest.mark.integration
def test_live_current_user_document_is_readable_without_mutation() -> None:
    service = _live_service()
    try:
        before_images = service.list_open_images()
        if not before_images:
            pytest.skip("Live GIMP is reachable but has no pre-existing image")
        before_context = service.get_current_context()
        current = before_context["current_image"]
        assert isinstance(current, dict)
        image_id = current["image_id"]
        assert before_context["open_image_count"] >= 1

        roots = service.list_layers(image_id)
        tree = service.get_layer_tree(image_id)
        selected = service.get_selected_layers(image_id)
        flattened = _flatten_tree(tree)
        assert [item["layer_id"] for item in selected] == before_context["selected_layer_ids"]
        assert {item["layer_id"] for item in flattened} >= {item["layer_id"] for item in roots}
        for item in flattened:
            assert service.get_layer_info(image_id, item["layer_id"])["image_id"] == image_id

        after_context = service.get_current_context()
        assert after_context["current_image"] == current
        assert after_context["selected_layer_ids"] == before_context["selected_layer_ids"]
        assert after_context["open_image_ids"] == before_context["open_image_ids"]
    finally:
        service.close()


@pytest.mark.integration
def test_live_nested_groups_and_multi_layer_selection_are_bounded_and_cleaned_up() -> None:
    service = _live_service()
    image_id: int | None = None
    try:
        image = service.create_image(32, 32)
        image_id = image["image_id"]
        outer = service.create_layer_group(image_id, "Outer group")
        child = service.create_layer(image_id, "Child layer", parent_id=outer["layer_id"])
        inner = service.create_layer_group(image_id, "Inner group", parent_id=outer["layer_id"])
        nested = service.create_layer(image_id, "Nested child", parent_id=inner["layer_id"])

        tree = service.get_layer_tree(image_id, max_depth=4, max_items=10)
        assert len(tree) == 1
        assert tree[0]["layer_id"] == outer["layer_id"]
        assert tree[0]["is_group"] is True
        assert {item["layer_id"] for item in tree[0]["children"]} == {
            child["layer_id"],
            inner["layer_id"],
        }
        nested_tree = next(
            item for item in tree[0]["children"] if item["layer_id"] == inner["layer_id"]
        )
        assert nested_tree["children"][0]["layer_id"] == nested["layer_id"]
        assert nested_tree["children"][0]["parent_id"] == inner["layer_id"]

        selected = service.set_selected_layers(image_id, [child["layer_id"], nested["layer_id"]])
        assert set(selected["selected_layer_ids"]) == {
            child["layer_id"],
            nested["layer_id"],
        }
        assert set(item["layer_id"] for item in service.get_selected_layers(image_id)) == set(
            selected["selected_layer_ids"]
        )
        with pytest.raises(RuntimeError, match="recursion-depth"):
            service.get_layer_tree(image_id, max_depth=0)
    finally:
        if image_id is not None:
            service.close_image(image_id, discard=True)
        service.close()


@pytest.mark.integration
def test_live_gimp_non_destructive_filters() -> None:
    service = _live_service()
    image_id: int | None = None
    try:
        image = service.create_image(16, 16)
        image_id = image["image_id"]
        layer = service.create_layer(image_id, "Live filter test")

        blur = service.apply_gaussian_blur_filter(
            layer["layer_id"], 1.5, name="Live Gaussian", opacity=100
        )
        brightness_contrast = service.apply_brightness_contrast_filter(
            layer["layer_id"], 0.2, 0.3, name="Live B/C", opacity=75
        )
        filters = service.list_drawable_filters(layer["layer_id"])

        assert blur["filter_id"] != brightness_contrast["filter_id"]
        assert blur["operation"] == "gegl:gaussian-blur"
        assert blur["name"] == "Live Gaussian"
        assert brightness_contrast["operation"] == "gegl:brightness-contrast"
        assert brightness_contrast["opacity"] == pytest.approx(0.75)
        assert {item["filter_id"] for item in filters} == {
            blur["filter_id"],
            brightness_contrast["filter_id"],
        }
        assert all(item["non_destructive"] is True for item in filters)

        with pytest.raises(ValueError):
            service.apply_gaussian_blur_filter(layer["layer_id"], float("nan"))
    finally:
        if image_id is not None:
            service.close_image(image_id, discard=True)
        service.close()


@pytest.mark.integration
def test_live_synthetic_layer_mask_and_metrics() -> None:
    service = _live_service()
    image_id: int | None = None
    try:
        image = service.create_image(32, 32)
        image_id = image["image_id"]
        layer = service.create_layer(image_id, "Synthetic mask subject")
        layer_id = layer["layer_id"]
        service.select_rectangle(image_id, 8, 8, 16, 16)
        info = service.create_layer_mask(image_id, layer_id, "selection")
        assert info["has_mask"] is True
        assert isinstance(info["mask_id"], int)
        samples = service._sample_mask(info["mask_id"], 32, 32, max_axis_samples=16)  # type: ignore[index]
        quality = mask_metrics(samples, 32, 32)
        assert quality["all_transparent"] is False
        assert quality["all_opaque"] is False
        assert quality["partial_alpha_ratio"] == 0
        assert quality["mask_coverage"] > 0
        disabled = service.set_layer_mask_enabled(image_id, layer_id, False)
        assert disabled["enabled"] is False
        enabled = service.set_layer_mask_enabled(image_id, layer_id, True)
        assert enabled["enabled"] is True
    finally:
        if image_id is not None:
            service.select_none(image_id)
            service.close_image(image_id, discard=True)
        service.close()
