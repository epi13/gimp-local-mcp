"""High-level GIMP operations composed over the transport and PDB gateway."""

from __future__ import annotations

import logging
import math
import shutil
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import Config
from .errors import PathPolicyError, ScriptFuError, UnsafeOperationError
from .gimp.filters import DrawableFilterGateway, DrawableFilterSpec
from .gimp.masks import LayerMaskGateway
from .gimp.metrics import mask_metrics
from .gimp.pdb import PdbCatalog, PdbInvoker
from .gimp.scheme import parse_scheme, unwrap
from .gimp.serializer import SchemeNull, SchemeSymbol, SchemeVector, scheme_call, with_v3
from .gimp.transport import ScriptFuClient
from .models import ImageInfo, LayerInfo, LayerNode
from .vision import VisionClient
from .vision.artifacts import (
    MaskArtifactError,
    complement_mask_png,
    mask_overlap_statistics,
    union_mask_png,
    validate_mask_png,
)
from .vision.models import SegmentationCandidate, SegmentationRequest, SegmentationResult
from .vision.refinement import IdentityMaskRefiner

logger = logging.getLogger(__name__)

_MAX_DECOMPOSITION_CONCEPTS = 8
_MAX_INSTANCES_PER_CONCEPT = 8
_MAX_GENERATED_VISION_LAYERS = 24


def _scalar(value: Any) -> Any:
    value = unwrap(value)
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _strict_object_ids(value: Any, label: str) -> list[int]:
    """Extract a GIMP object array without silently dropping malformed members."""

    value = unwrap(value)
    if not isinstance(value, list):
        raise RuntimeError(f"GIMP returned malformed {label}: expected an object array")
    if len(value) >= 2 and isinstance(value[1], list):
        values = value[1]
    elif len(value) == 1 and isinstance(value[0], list):
        values = value[0]
    else:
        values = value
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in values):
        raise RuntimeError(
            f"GIMP returned malformed {label}: object IDs must be non-negative integers"
        )
    return [int(item) for item in values]


class GimpService:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.from_env()
        self.client = ScriptFuClient(self.config)
        self.catalog = PdbCatalog(self.client)
        self.pdb = PdbInvoker(self.client, self.catalog)
        self.filters = DrawableFilterGateway(self.evaluate)
        self.masks = LayerMaskGateway(self.evaluate)
        self.vision = VisionClient(self.config)
        self.mask_refiner = IdentityMaskRefiner()

    def close(self) -> None:
        self.vision.close()
        self.client.close()

    def vision_status(self) -> dict[str, Any]:
        """Report local provider capability without exposing worker internals."""

        return self.vision.capabilities().as_dict()

    def evaluate(self, expression: str) -> Any:
        """Internal evaluation primitive; no MCP tool exposes this method."""

        return unwrap(parse_scheme(self.client.execute(with_v3(expression))))

    def status(self) -> dict[str, Any]:
        version = _scalar(self.evaluate(scheme_call("gimp-version")))
        images = self.list_open_images()
        return {
            "connected": True,
            "gimp_version": version,
            "configured_endpoint": f"{self.config.host}:{self.config.port}",
            "open_image_count": len(images),
            "local_only": not self.config.allow_remote,
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "gimp_version": _scalar(self.evaluate(scheme_call("gimp-version"))),
            "transport": "Script-Fu TCP",
            "pdb_discovery": [
                "search",
                "exists",
                "procedure counts",
                "documentation",
                "typed metadata capability states",
            ],
            "high_level_tools": [
                "images",
                "layers",
                "transforms",
                "selections",
                "undo",
                "basic adjustments",
                "non-destructive GEGL filters",
                "current document context",
                "recursive layer trees",
                "multi-layer selection inspection",
                "multi-layer selection control",
                "reusable layer masks",
                "bounded high-key subject isolation",
                "optional local semantic vision status and segmentation",
                "non-destructive semantic subject isolation",
                "persistent complementary subject/background layer decomposition",
                "bounded multi-concept decomposition with overlap reporting",
            ],
            "limitations": [
                "The Script-Fu server may not expose a default display; with one open image, "
                "current-image resolution uses an explicit single-image fallback.",
                "The default Script-Fu adapter reports argument GParamSpec metadata as "
                "unavailable; no signatures are guessed.",
                "Export metadata options are left to GIMP's configured defaults in v0.1.",
                "Foreground extraction is unavailable through the tested Script-Fu bridge;\n"
                "subject isolation uses a border-seeded contiguous-color fallback with\n"
                "observable matte proxies rather than semantic accuracy.",
                "Semantic vision is optional and out-of-process; the configured worker must\n"
                "report text segmentation capability before auto mode selects it.",
            ],
        }

    def list_open_images(self) -> list[dict[str, Any]]:
        result = self.evaluate(scheme_call("gimp-get-images"))
        return [
            self.get_image_info(image_id) for image_id in _strict_object_ids(result, "image list")
        ]

    def get_active_image(self) -> dict[str, Any] | None:
        image, _source, _display_id = self._resolve_current_image(self.list_open_images())
        return image

    def get_current_context(self) -> dict[str, Any]:
        """Return a concise, evidence-backed snapshot for the next editing decision."""

        images = self.list_open_images()
        image, source, display_id = self._resolve_current_image(images)
        selected_layers = self.get_selected_layers(image["image_id"]) if image else []
        return {
            "gimp_version": _scalar(self.evaluate(scheme_call("gimp-version"))),
            "configured_endpoint": f"{self.config.host}:{self.config.port}",
            "open_image_count": len(images),
            "open_image_ids": [item["image_id"] for item in images],
            "current_image": image,
            "current_image_source": source,
            "default_display_id": display_id,
            "selection_model": "multi-layer",
            "selected_layer_ids": [item["layer_id"] for item in selected_layers],
            "selected_layers": selected_layers,
        }

    def _resolve_current_image(
        self, images: list[dict[str, Any]]
    ) -> tuple[dict[str, Any] | None, str, int | None]:
        """Resolve a display image when possible without inventing an active image."""

        display_id: int | None = None
        try:
            display = _scalar(self.evaluate(scheme_call("gimp-default-display")))
            if isinstance(display, int) and display >= 0:
                display_id = display
                image_id = _scalar(self.evaluate(scheme_call("gimp-display-get-image", [display])))
                if isinstance(image_id, int) and image_id >= 0:
                    for image in images:
                        if image["image_id"] == image_id:
                            return image, "default-display", display_id
        except ScriptFuError:
            # Some GIMP 3.x Script-Fu server requests have no menu display context.
            pass
        if not images:
            return None, "none-open", display_id
        if len(images) == 1:
            return images[0], "single-open-image", display_id
        return None, "ambiguous-multiple-images", display_id

    def get_image_info(self, image_id: int) -> dict[str, Any]:
        self._id(image_id, "image_id")
        values = unwrap(
            self.evaluate(
                "(list "
                f"(gimp-image-get-name {image_id}) "
                f"(gimp-image-get-width {image_id}) "
                f"(gimp-image-get-height {image_id}) "
                f"(gimp-image-get-base-type {image_id}) "
                f"(gimp-image-get-file {image_id}) "
                f"(gimp-image-is-dirty {image_id}))"
            )
        )
        values = values if isinstance(values, list) else []
        info = ImageInfo(
            image_id=image_id,
            name=values[0] if len(values) > 0 and isinstance(values[0], str) else None,
            width=int(values[1]) if len(values) > 1 and isinstance(values[1], int) else 0,
            height=int(values[2]) if len(values) > 2 and isinstance(values[2], int) else 0,
            base_type=values[3] if len(values) > 3 else None,
            file=self._file_value(values[4] if len(values) > 4 else None),
            dirty=values[5] if len(values) > 5 and isinstance(values[5], bool) else None,
        )
        return info.as_dict()

    def open_image(self, path: str) -> dict[str, Any]:
        source = self._existing_file(path)
        image_id = _scalar(
            self.evaluate(
                scheme_call("gimp-file-load", [SchemeSymbol("RUN-NONINTERACTIVE"), str(source)])
            )
        )
        if not isinstance(image_id, int) or image_id < 0:
            raise RuntimeError(f"GIMP did not return an image ID when opening {source}")
        return self.get_image_info(image_id)

    def create_image(self, width: int, height: int, base_type: str = "RGB") -> dict[str, Any]:
        self._size(width, height)
        normalized_type = base_type.upper()
        if normalized_type not in {"RGB", "GRAY", "INDEXED"}:
            raise ValueError("base_type must be RGB, GRAY, or INDEXED")
        enum = SchemeSymbol(normalized_type)
        image_id = _scalar(self.evaluate(scheme_call("gimp-image-new", [width, height, enum])))
        if not isinstance(image_id, int) or image_id < 0:
            raise RuntimeError("GIMP did not return an image ID when creating an image")
        return self.get_image_info(image_id)

    def save_xcf(self, image_id: int, path: str, overwrite: bool = False) -> dict[str, Any]:
        target = self._output_file(path, overwrite)
        if target.suffix.lower() != ".xcf":
            raise PathPolicyError("save_xcf paths must use the .xcf extension")
        self._save(image_id, target)
        return {"image_id": image_id, "path": str(target), "format": "xcf"}

    def export_image(self, image_id: int, path: str, overwrite: bool = False) -> dict[str, Any]:
        target = self._output_file(path, overwrite)
        if not target.suffix:
            raise PathPolicyError("export_image paths must include a format extension")
        self._export(image_id, target)
        return {
            "image_id": image_id,
            "path": str(target),
            "format": target.suffix.lower().lstrip(".") or None,
        }

    def close_image(self, image_id: int, discard: bool = False) -> dict[str, Any]:
        self._id(image_id, "image_id")
        if not discard:
            dirty = self._scalar_bool(self.evaluate(scheme_call("gimp-image-is-dirty", [image_id])))
            if dirty:
                raise UnsafeOperationError(
                    "Image has unsaved changes; save it first or call close_image with discard=true"
                )
        self.evaluate(scheme_call("gimp-image-delete", [image_id]))
        return {"image_id": image_id, "closed": True, "discarded": discard}

    def list_layers(self, image_id: int) -> list[dict[str, Any]]:
        self._id(image_id, "image_id")
        ids = _strict_object_ids(
            self.evaluate(scheme_call("gimp-image-get-layers", [image_id])), "root layer list"
        )
        return [
            self.get_layer_info(image_id, layer_id, index) for index, layer_id in enumerate(ids)
        ]

    def get_layer_info(
        self, image_id: int, layer_id: int, position: int | None = None
    ) -> dict[str, Any]:
        self._id(image_id, "image_id")
        self._id(layer_id, "layer_id")
        values = unwrap(
            self.evaluate(
                "(list "
                f"(gimp-item-get-name {layer_id}) "
                f"(gimp-drawable-get-width {layer_id}) "
                f"(gimp-drawable-get-height {layer_id}) "
                f"(gimp-item-get-visible {layer_id}) "
                f"(gimp-layer-get-opacity {layer_id}) "
                f"(gimp-layer-get-mode {layer_id}) "
                f"(gimp-item-get-parent {layer_id}) "
                f"(gimp-item-is-group {layer_id}) "
                f"(gimp-item-get-image {layer_id}))"
            )
        )
        values = values if isinstance(values, list) else []
        parent = (
            values[6] if len(values) > 6 and isinstance(values[6], int) and values[6] >= 0 else None
        )
        is_group = values[7] if len(values) > 7 and isinstance(values[7], bool) else None
        owner = values[8] if len(values) > 8 and isinstance(values[8], int) else None
        if owner is not None and owner != image_id:
            raise ValueError(f"Layer {layer_id} belongs to image {owner}, not image {image_id}")
        info = LayerInfo(
            layer_id=layer_id,
            image_id=image_id,
            name=values[0] if len(values) > 0 and isinstance(values[0], str) else None,
            width=int(values[1]) if len(values) > 1 and isinstance(values[1], int) else 0,
            height=int(values[2]) if len(values) > 2 and isinstance(values[2], int) else 0,
            visible=values[3] if len(values) > 3 and isinstance(values[3], bool) else None,
            opacity=values[4] if len(values) > 4 and isinstance(values[4], (int, float)) else None,
            mode=values[5] if len(values) > 5 else None,
            position=position if position is not None else self._item_position(image_id, layer_id),
            parent_id=parent,
            is_group=is_group,
        )
        return info.as_dict()

    def get_layer_tree(
        self, image_id: int, *, max_depth: int = 32, max_items: int = 1000
    ) -> list[dict[str, Any]]:
        """Return the complete bounded root-to-leaf layer/group hierarchy."""

        self._id(image_id, "image_id")
        if not isinstance(max_depth, int) or not 0 <= max_depth <= 128:
            raise ValueError("max_depth must be an integer from 0 to 128")
        if not isinstance(max_items, int) or not 1 <= max_items <= 10_000:
            raise ValueError("max_items must be an integer from 1 to 10000")
        roots = self.list_layers(image_id)
        seen: set[int] = set()
        count = 0

        def build(info: dict[str, Any], depth: int) -> LayerNode:
            nonlocal count
            layer_id = info["layer_id"]
            if layer_id in seen:
                raise RuntimeError(f"GIMP returned a cyclic layer hierarchy at layer {layer_id}")
            if depth > max_depth:
                raise RuntimeError("GIMP layer hierarchy exceeds the recursion-depth limit")
            count += 1
            if count > max_items:
                raise RuntimeError("GIMP layer hierarchy exceeds the item-count limit")
            if info.get("is_group") is None:
                raise RuntimeError(f"GIMP did not report whether layer {layer_id} is a group")
            if depth == 0 and info.get("parent_id") is not None:
                raise RuntimeError(
                    f"GIMP reported a parent for root layer {layer_id}: {info['parent_id']}"
                )
            seen.add(layer_id)
            children: list[LayerNode] = []
            if info.get("is_group") is True:
                child_ids = _strict_object_ids(
                    self.evaluate(scheme_call("gimp-item-get-children", [layer_id])),
                    f"children of layer {layer_id}",
                )
                for position, child_id in enumerate(child_ids):
                    child_info = self.get_layer_info(image_id, child_id, position)
                    if child_info.get("parent_id") != layer_id:
                        raise RuntimeError(
                            f"GIMP reported inconsistent parent for layer {child_id}: "
                            f"expected {layer_id}, got {child_info.get('parent_id')}"
                        )
                    children.append(build(child_info, depth + 1))
            seen.remove(layer_id)
            return LayerNode(LayerInfo(**info), tuple(children))

        return [build(info, 0).as_dict() for info in roots]

    def get_selected_layers(self, image_id: int) -> list[dict[str, Any]]:
        """Return all currently selected layers, preserving GIMP's multi-select model."""

        self._id(image_id, "image_id")
        ids = _strict_object_ids(
            self.evaluate(scheme_call("gimp-image-get-selected-layers", [image_id])),
            "selected layer list",
        )
        return [self.get_layer_info(image_id, layer_id) for layer_id in ids]

    def set_selected_layers(self, image_id: int, layer_ids: list[int]) -> dict[str, Any]:
        """Set and read back GIMP's multi-layer selection state."""

        self._id(image_id, "image_id")
        if not isinstance(layer_ids, list) or not layer_ids:
            raise ValueError("layer_ids must be a non-empty list")
        if len(layer_ids) > 1000:
            raise ValueError("layer_ids cannot contain more than 1000 layers")
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("layer_ids cannot contain duplicates")
        for layer_id in layer_ids:
            self._id(layer_id, "layer_id")
            self._assert_item_belongs_to_image(image_id, layer_id)
        self.evaluate(
            scheme_call(
                "gimp-image-set-selected-layers",
                [image_id, SchemeVector(tuple(layer_ids))],
            )
        )
        selected = self.get_selected_layers(image_id)
        selected_ids = [item["layer_id"] for item in selected]
        if set(selected_ids) != set(layer_ids):
            raise RuntimeError(
                f"GIMP selected-layer read-back differed: requested {layer_ids}, got {selected_ids}"
            )
        return {
            "image_id": image_id,
            "selection_model": "multi-layer",
            "selected_layer_ids": selected_ids,
            "selected_layers": selected,
        }

    def create_layer(
        self,
        image_id: int,
        name: str,
        width: int | None = None,
        height: int | None = None,
        opacity: float = 100.0,
        parent_id: int | None = None,
    ) -> dict[str, Any]:
        self._id(image_id, "image_id")
        if not name:
            raise ValueError("Layer name cannot be empty")
        self._validate_parent(image_id, parent_id)
        image = self.get_image_info(image_id)
        layer_width = image["width"] if width is None else width
        layer_height = image["height"] if height is None else height
        self._size(layer_width, layer_height)
        if not 0 <= opacity <= 100:
            raise ValueError("Layer opacity must be between 0 and 100")
        layer_expr = scheme_call(
            "gimp-layer-new",
            [
                image_id,
                name,
                layer_width,
                layer_height,
                SchemeSymbol("RGBA-IMAGE"),
                opacity,
                SchemeSymbol("LAYER-MODE-NORMAL"),
            ],
        )
        expression = (
            f"(let ((layer {layer_expr})) (gimp-image-insert-layer {image_id} layer "
            f"{parent_id if parent_id is not None else -1} 0) layer)"
        )
        layer_id = _scalar(self._run_undo_group(image_id, expression))
        if not isinstance(layer_id, int) or layer_id < 0:
            raise RuntimeError("GIMP did not return a layer ID when creating a layer")
        return self.get_layer_info(image_id, layer_id, 0)

    def create_layer_group(
        self, image_id: int, name: str, parent_id: int | None = None
    ) -> dict[str, Any]:
        """Create a group layer at the top of the requested layer level."""

        self._id(image_id, "image_id")
        if not name:
            raise ValueError("Layer group name cannot be empty")
        self._validate_parent(image_id, parent_id)
        group_expr = scheme_call("gimp-group-layer-new", [image_id, name])
        expression = (
            f"(let ((group {group_expr})) (gimp-image-insert-layer {image_id} group "
            f"{parent_id if parent_id is not None else -1} 0) group)"
        )
        group_id = _scalar(self._run_undo_group(image_id, expression))
        if not isinstance(group_id, int) or group_id < 0:
            raise RuntimeError("GIMP did not return a layer-group ID")
        return self.get_layer_info(image_id, group_id, 0)

    def duplicate_layer(self, image_id: int, layer_id: int) -> dict[str, Any]:
        self._id(image_id, "image_id")
        self._id(layer_id, "layer_id")
        expression = (
            f"(let ((copy (gimp-layer-copy {layer_id} #t))) "
            f"(gimp-image-insert-layer {image_id} copy -1 0) copy)"
        )
        new_id = _scalar(self._run_undo_group(image_id, expression))
        if not isinstance(new_id, int) or new_id < 0:
            raise RuntimeError("GIMP did not return a layer ID when duplicating a layer")
        return self.get_layer_info(image_id, new_id, 0)

    def rename_layer(self, layer_id: int, name: str) -> dict[str, Any]:
        self._id(layer_id, "layer_id")
        if not name:
            raise ValueError("Layer name cannot be empty")
        self.evaluate(scheme_call("gimp-item-set-name", [layer_id, name]))
        return {"layer_id": layer_id, "name": name}

    def delete_layer(self, image_id: int, layer_id: int) -> dict[str, Any]:
        self._id(image_id, "image_id")
        self._id(layer_id, "layer_id")
        self.evaluate(scheme_call("gimp-image-remove-layer", [image_id, layer_id]))
        return {"image_id": image_id, "layer_id": layer_id, "deleted": True}

    def set_layer_visibility(self, layer_id: int, visible: bool) -> dict[str, Any]:
        self.evaluate(scheme_call("gimp-item-set-visible", [layer_id, visible]))
        return {"layer_id": layer_id, "visible": visible}

    def set_layer_opacity(self, layer_id: int, opacity: float) -> dict[str, Any]:
        if not 0 <= opacity <= 100:
            raise ValueError("Layer opacity must be between 0 and 100")
        self.evaluate(scheme_call("gimp-layer-set-opacity", [layer_id, opacity]))
        return {"layer_id": layer_id, "opacity": opacity}

    def set_layer_mode(self, layer_id: int, mode: str) -> dict[str, Any]:
        self.evaluate(scheme_call("gimp-layer-set-mode", [layer_id, SchemeSymbol(mode.upper())]))
        return {"layer_id": layer_id, "mode": mode.upper()}

    def move_layer(
        self, image_id: int, layer_id: int, position: int, parent_id: int | None = None
    ) -> dict[str, Any]:
        self._id(image_id, "image_id")
        self._id(layer_id, "layer_id")
        self._assert_item_belongs_to_image(image_id, layer_id)
        self._validate_parent(image_id, parent_id)
        if position < 0:
            raise ValueError("Layer position cannot be negative")
        parent = SchemeNull() if parent_id is None else parent_id
        self.evaluate(
            scheme_call("gimp-image-reorder-item", [image_id, layer_id, parent, position])
        )
        return self.get_layer_info(image_id, layer_id, position)

    def merge_down(
        self, image_id: int, layer_id: int, merge_type: str = "EXPAND-AS-NECESSARY"
    ) -> dict[str, Any]:
        merged = _scalar(
            self.evaluate(
                scheme_call(
                    "gimp-image-merge-down", [image_id, layer_id, SchemeSymbol(merge_type.upper())]
                )
            )
        )
        return {"image_id": image_id, "layer_id": layer_id, "merged_layer_id": merged}

    def resize_image(self, image_id: int, width: int, height: int) -> dict[str, Any]:
        self._size(width, height)
        self.evaluate(scheme_call("gimp-image-scale", [image_id, width, height]))
        return self.get_image_info(image_id)

    def resize_canvas(
        self, image_id: int, width: int, height: int, offset_x: int = 0, offset_y: int = 0
    ) -> dict[str, Any]:
        self._size(width, height)
        self.evaluate(
            scheme_call("gimp-image-resize", [image_id, width, height, offset_x, offset_y])
        )
        return self.get_image_info(image_id)

    def crop_image(
        self, image_id: int, width: int, height: int, offset_x: int = 0, offset_y: int = 0
    ) -> dict[str, Any]:
        self._size(width, height)
        self.evaluate(scheme_call("gimp-image-crop", [image_id, width, height, offset_x, offset_y]))
        return self.get_image_info(image_id)

    def rotate(self, image_id: int, rotation: str) -> dict[str, Any]:
        self.evaluate(scheme_call("gimp-image-rotate", [image_id, SchemeSymbol(rotation.upper())]))
        return self.get_image_info(image_id)

    def flip(self, image_id: int, direction: str) -> dict[str, Any]:
        self.evaluate(scheme_call("gimp-image-flip", [image_id, SchemeSymbol(direction.upper())]))
        return self.get_image_info(image_id)

    def select_all(self, image_id: int) -> dict[str, Any]:
        self.evaluate(scheme_call("gimp-selection-all", [image_id]))
        return {"image_id": image_id, "selection": "all"}

    def select_none(self, image_id: int) -> dict[str, Any]:
        self.evaluate(scheme_call("gimp-selection-none", [image_id]))
        return {"image_id": image_id, "selection": "none"}

    def invert_selection(self, image_id: int) -> dict[str, Any]:
        self.evaluate(scheme_call("gimp-selection-invert", [image_id]))
        return {"image_id": image_id, "selection": "inverted"}

    def select_rectangle(
        self, image_id: int, x: int, y: int, width: int, height: int
    ) -> dict[str, Any]:
        self._size(width, height)
        self.evaluate(
            scheme_call(
                "gimp-image-select-rectangle",
                [image_id, SchemeSymbol("CHANNEL-OP-REPLACE"), x, y, width, height],
            )
        )
        return {
            "image_id": image_id,
            "shape": "rectangle",
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        }

    def select_ellipse(
        self, image_id: int, x: int, y: int, width: int, height: int
    ) -> dict[str, Any]:
        self._size(width, height)
        self.evaluate(
            scheme_call(
                "gimp-image-select-ellipse",
                [image_id, SchemeSymbol("CHANNEL-OP-REPLACE"), x, y, width, height],
            )
        )
        return {
            "image_id": image_id,
            "shape": "ellipse",
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        }

    def select_layer_alpha(self, image_id: int, layer_id: int) -> dict[str, Any]:
        self._id(image_id, "image_id")
        self._id(layer_id, "layer_id")
        self._assert_item_belongs_to_image(image_id, layer_id)
        self.evaluate(
            scheme_call(
                "gimp-image-select-item", [image_id, SchemeSymbol("CHANNEL-OP-REPLACE"), layer_id]
            )
        )
        return {"image_id": image_id, "layer_id": layer_id, "selection": "layer-alpha"}

    def get_layer_mask_info(self, image_id: int, layer_id: int) -> dict[str, Any]:
        self._id(image_id, "image_id")
        self._id(layer_id, "layer_id")
        self._assert_item_belongs_to_image(image_id, layer_id)
        return self.masks.get(layer_id).as_dict()

    def create_layer_mask(
        self,
        image_id: int,
        layer_id: int,
        mask_type: str = "selection",
    ) -> dict[str, Any]:
        self._id(image_id, "image_id")
        self._id(layer_id, "layer_id")
        self._assert_item_belongs_to_image(image_id, layer_id)
        return self.masks.create_and_attach(layer_id, mask_type).as_dict()

    def set_layer_mask_enabled(self, image_id: int, layer_id: int, enabled: bool) -> dict[str, Any]:
        self._id(image_id, "image_id")
        self._id(layer_id, "layer_id")
        self._assert_item_belongs_to_image(image_id, layer_id)
        return self.masks.set_enabled(layer_id, enabled).as_dict()

    def _selection_snapshot(self, image_id: int) -> dict[str, Any]:
        values = unwrap(self.evaluate(scheme_call("gimp-selection-bounds", [image_id])))
        if not isinstance(values, list) or len(values) < 5 or not isinstance(values[0], bool):
            raise RuntimeError(f"GIMP returned malformed selection bounds: {values!r}")
        return {
            "active": values[0],
            "x": values[1] if isinstance(values[1], int) else None,
            "y": values[2] if isinstance(values[2], int) else None,
            "width": values[3] if isinstance(values[3], int) else None,
            "height": values[4] if isinstance(values[4], int) else None,
        }

    @staticmethod
    def _border_points(width: int, height: int) -> list[tuple[int, int]]:
        fractions = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
        points = [(round((width - 1) * fraction), 0) for fraction in fractions]
        points += [(round((width - 1) * fraction), height - 1) for fraction in fractions]
        points += [(0, round((height - 1) * fraction)) for fraction in fractions]
        points += [(width - 1, round((height - 1) * fraction)) for fraction in fractions]
        return list(dict.fromkeys(points))

    def _sample_pixel(self, drawable_id: int, x: int, y: int) -> list[int]:
        value = unwrap(self.evaluate(scheme_call("gimp-drawable-get-pixel", [drawable_id, x, y])))
        if not isinstance(value, list) or not 1 <= len(value) <= 4:
            raise RuntimeError(f"GIMP returned malformed pixel at {x},{y}: {value!r}")
        pixel: list[int] = []
        for channel in value:
            if not isinstance(channel, (int, float)) or isinstance(channel, bool):
                raise RuntimeError(f"GIMP returned malformed pixel channel: {value!r}")
            if not 0 <= float(channel) <= 255 or not math.isfinite(float(channel)):
                raise RuntimeError(f"GIMP returned out-of-range pixel channel: {value!r}")
            pixel.append(int(round(float(channel))))
        return pixel

    def _border_characteristics(self, layer_id: int, width: int, height: int) -> dict[str, Any]:
        samples = []
        for x, y in self._border_points(width, height):
            pixel = self._sample_pixel(layer_id, x, y)
            luminance = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
            samples.append({"x": x, "y": y, "rgb": pixel[:3], "luminance": luminance})
        luminances = [item["luminance"] for item in samples]
        high_key = [value for value in luminances if value >= 170]
        mean = sum(luminances) / len(luminances) if luminances else 0.0
        spread = max(luminances) - min(luminances) if luminances else 0.0
        return {
            "sample_count": len(samples),
            "high_key_sample_count": len(high_key),
            "high_key_ratio": len(high_key) / len(samples) if samples else 0.0,
            "mean_luminance": round(mean, 3),
            "luminance_spread": round(spread, 3),
            "samples": samples[:32],
        }

    def _select_border_background(
        self,
        image_id: int,
        layer_id: int,
        width: int,
        height: int,
        *,
        threshold: int,
        feather: int,
        minimum_luminance: float | None = 170,
    ) -> dict[str, Any]:
        self.evaluate(scheme_call("gimp-selection-none", [image_id]))
        self.evaluate(scheme_call("gimp-context-set-sample-threshold-int", [threshold]))
        selected_points = []
        for x, y in self._border_points(width, height):
            pixel = self._sample_pixel(layer_id, x, y)
            luminance = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
            if minimum_luminance is not None and luminance < minimum_luminance:
                continue
            selected_points.append((x, y))
            operation = "CHANNEL-OP-REPLACE" if len(selected_points) == 1 else "CHANNEL-OP-ADD"
            self.evaluate(
                scheme_call(
                    "gimp-image-select-contiguous-color",
                    [image_id, SchemeSymbol(operation), layer_id, x, y],
                )
            )
        if not selected_points:
            raise RuntimeError("no sufficiently light perimeter samples were available")
        if feather:
            self.evaluate(scheme_call("gimp-selection-feather", [image_id, feather]))
        self.evaluate(scheme_call("gimp-selection-invert", [image_id]))
        return {
            "selected_border_points": len(selected_points),
            "threshold": threshold,
            "feather": feather,
            "minimum_luminance": minimum_luminance,
        }

    def _sample_mask(
        self, mask_id: int, width: int, height: int, *, max_axis_samples: int = 16
    ) -> list[tuple[int, int, int]]:
        step = max(1, math.ceil(max(width, height) / max_axis_samples))
        xs = sorted(set(range(0, width, step)) | {width - 1})
        ys = sorted(set(range(0, height, step)) | {height - 1})
        samples: list[tuple[int, int, int]] = []
        for y in ys:
            for x in xs:
                pixel = self._sample_pixel(mask_id, x, y)
                if len(pixel) != 1:
                    raise RuntimeError(f"GIMP returned non-grayscale mask pixel: {pixel!r}")
                samples.append((x, y, self.masks.validate_alpha(pixel[0])))
        return samples

    @contextmanager
    def _vision_snapshot(self, image_id: int) -> Iterator[tuple[Path, Path]]:
        """Save a duplicate GIMP image, never the user's image, to a temp PNG."""

        temporary_directory = Path(tempfile.mkdtemp(prefix="gimp-local-mcp-vision-"))
        duplicate_id: int | None = None
        snapshot = temporary_directory / "input.png"
        try:
            duplicate_id = _scalar(self.evaluate(scheme_call("gimp-image-duplicate", [image_id])))
            if not isinstance(duplicate_id, int) or duplicate_id < 0:
                raise RuntimeError("GIMP did not return a temporary duplicate image ID")
            self._save(duplicate_id, snapshot)
            if not snapshot.is_file():
                raise RuntimeError("GIMP did not create the temporary vision snapshot")
            yield snapshot, temporary_directory
        finally:
            if duplicate_id is not None:
                try:
                    self.close_image(duplicate_id, discard=True)
                except Exception:
                    logger.exception("failed to close temporary vision image %s", duplicate_id)
            try:
                shutil.rmtree(temporary_directory)
            except OSError:
                logger.exception(
                    "failed to clean temporary vision snapshot %s", temporary_directory
                )

    @staticmethod
    def _vision_request(
        image_path: Path,
        output_directory: Path,
        prompt: str,
        *,
        max_candidates: int,
        minimum_score: float,
        mode: str,
    ) -> SegmentationRequest:
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.strip()) > 256:
            raise ValueError("prompt must be a non-empty string of at most 256 characters")
        return SegmentationRequest(
            image_path=image_path,
            prompt=prompt.strip(),
            max_candidates=max_candidates,
            minimum_score=minimum_score,
            mode=mode,
            output_directory=output_directory,
        )

    def _validate_vision_result(
        self,
        result: SegmentationResult,
        output_directory: Path,
        width: int,
        height: int,
        minimum_score: float = 0.0,
    ) -> SegmentationResult:
        if not result.candidates:
            raise RuntimeError("vision provider returned no segmentation candidates")
        checked = []
        root = output_directory.resolve()
        for candidate in result.candidates:
            artifact_path = candidate.mask.path.resolve()
            if root not in artifact_path.parents:
                raise RuntimeError("vision provider mask escaped the temporary artifact directory")
            try:
                artifact = validate_mask_png(artifact_path)
            except (OSError, MaskArtifactError) as exc:
                raise RuntimeError(
                    f"vision provider returned an invalid mask artifact: {exc}"
                ) from exc
            if artifact.width != width or artifact.height != height:
                raise RuntimeError("vision provider mask dimensions do not match the source layer")
            checked.append(
                type(candidate)(
                    candidate.candidate_id,
                    candidate.concept,
                    candidate.score,
                    candidate.bounding_box,
                    artifact,
                    candidate.width,
                    candidate.height,
                    candidate.metadata,
                )
            )
        checked = [
            candidate
            for candidate in checked
            if candidate.score is None or candidate.score >= minimum_score
        ]
        if not checked:
            raise RuntimeError("vision provider returned no candidate meeting minimum_score")
        return type(result)(
            result.provider,
            result.model,
            tuple(checked),
            result.runtime_seconds,
            result.warnings,
            result.provenance,
            any(candidate.mask.soft_alpha for candidate in checked),
        )

    @staticmethod
    def _local_mask_quality(path: Path, width: int, height: int) -> dict[str, Any]:
        from .vision.artifacts import read_mask_png

        _artifact, pixels = read_mask_png(path)
        step = max(1, math.ceil(max(width, height) / 16))
        xs = sorted(set(range(0, width, step)) | {width - 1})
        ys = sorted(set(range(0, height, step)) | {height - 1})
        samples = [(x, y, pixels[y * width + x]) for y in ys for x in xs]
        return mask_metrics(samples, width, height)

    @staticmethod
    def _public_vision_result(
        result: SegmentationResult, width: int, height: int
    ) -> dict[str, Any]:
        candidates = []
        for candidate in result.candidates:
            candidates.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "concept": candidate.concept,
                    "score": candidate.score,
                    "bounding_box": candidate.bounding_box.as_dict()
                    if candidate.bounding_box
                    else None,
                    "width": candidate.width,
                    "height": candidate.height,
                    "mask_summary": {
                        "width": candidate.mask.width,
                        "height": candidate.mask.height,
                        "soft_alpha": candidate.mask.soft_alpha,
                        "quality_proxies": GimpService._local_mask_quality(
                            candidate.mask.path, width, height
                        ),
                        "artifact_retained": False,
                    },
                }
            )
        return {
            "provider": result.provider,
            "model": result.model,
            "candidates": candidates,
            "runtime_seconds": result.runtime_seconds,
            "warnings": list(result.warnings),
            "provenance": result.provenance,
            "soft_alpha": result.soft_alpha,
        }

    def segment_subject(
        self,
        image_id: int,
        layer_id: int,
        prompt: str,
        *,
        max_candidates: int = 3,
        minimum_score: float = 0.0,
        mode: str = "semantic",
    ) -> dict[str, Any]:
        """Segment a prompt from a temporary current-GIMP snapshot without mutation."""

        self._id(image_id, "image_id")
        self._id(layer_id, "layer_id")
        self._assert_item_belongs_to_image(image_id, layer_id)
        source = self.get_layer_info(image_id, layer_id)
        with self._vision_snapshot(image_id) as (snapshot, output_directory):
            request = self._vision_request(
                snapshot,
                output_directory,
                prompt,
                max_candidates=max_candidates,
                minimum_score=minimum_score,
                mode=mode,
            )
            result = self._validate_vision_result(
                self.vision.segment(request),
                output_directory,
                source["width"],
                source["height"],
                minimum_score,
            )
            return {
                "status": "completed",
                "image_id": image_id,
                "source_layer_id": layer_id,
                "prompt": prompt.strip(),
                "result": self._public_vision_result(result, source["width"], source["height"]),
                "source_preserved": True,
                "limitations": [
                    "Provider confidence is not semantic segmentation accuracy.",
                    "No ground-truth fox alpha matte is available for this image.",
                ],
            }

    def _apply_vision_mask(
        self,
        image_id: int,
        layer_id: int,
        mask_path: Path,
        width: int,
        height: int,
        bridge_directory: Path,
    ) -> dict[str, Any]:
        """Load a temporary RGBA alpha bridge and attach it through LayerMaskGateway."""

        from .vision.artifacts import read_mask_png, write_rgba_mask_png

        artifact, pixels = read_mask_png(mask_path)
        if artifact.width != width or artifact.height != height:
            raise RuntimeError("vision mask dimensions do not match the working layer")
        bridge_path = write_rgba_mask_png(
            bridge_directory / "mask-alpha.png", width, height, pixels
        )
        temporary_layer_id: int | None = None
        try:
            temporary_layer_id = _scalar(
                self.evaluate(
                    scheme_call(
                        "gimp-file-load-layer",
                        [SchemeSymbol("RUN-NONINTERACTIVE"), image_id, str(bridge_path)],
                    )
                )
            )
            if not isinstance(temporary_layer_id, int) or temporary_layer_id < 0:
                raise RuntimeError("GIMP did not return the temporary vision mask layer")
            self.evaluate(
                scheme_call("gimp-image-insert-layer", [image_id, temporary_layer_id, -1, 0])
            )
            self.select_layer_alpha(image_id, temporary_layer_id)
            mask = self.masks.create_and_attach(layer_id, "selection")
            if mask.mask_id is None:
                raise RuntimeError("GIMP did not return the attached vision mask identity")
            return mask.as_dict()
        finally:
            if temporary_layer_id is not None:
                try:
                    self.delete_layer(image_id, temporary_layer_id)
                except Exception:
                    logger.exception("failed to remove temporary vision mask layer")
            self.evaluate(scheme_call("gimp-selection-none", [image_id]))

    def isolate_subject_vision(
        self,
        image_id: int,
        layer_id: int,
        prompt: str,
        *,
        max_candidates: int = 3,
        minimum_score: float = 0.0,
        mode: str = "semantic",
    ) -> dict[str, Any]:
        """Duplicate a source layer and apply a local semantic mask non-destructively."""

        self._id(image_id, "image_id")
        self._id(layer_id, "layer_id")
        self._assert_item_belongs_to_image(image_id, layer_id)
        source = self.get_layer_info(image_id, layer_id)
        if self.masks.get(layer_id).has_mask:
            raise ValueError("source layer already has a mask; choose an unmasked source layer")
        if self._selection_snapshot(image_id)["active"]:
            raise ValueError(
                "isolate_subject_vision refuses to replace an existing image selection"
            )
        source_visibility = source.get("visible")
        started = time.perf_counter()
        working_layer_id: int | None = None
        with self._vision_snapshot(image_id) as (snapshot, output_directory):
            request = self._vision_request(
                snapshot,
                output_directory,
                prompt,
                max_candidates=max_candidates,
                minimum_score=minimum_score,
                mode=mode,
            )
            result = self._validate_vision_result(
                self.vision.segment(request),
                output_directory,
                source["width"],
                source["height"],
                minimum_score,
            )
            candidate = max(
                result.candidates, key=lambda item: item.score if item.score is not None else 0.0
            )
            refined = self.mask_refiner.refine(candidate.mask)
            try:
                working = self.duplicate_layer(image_id, layer_id)
                working_layer_id = working["layer_id"]
                label = prompt.strip().replace("\n", " ")[:96]
                self.rename_layer(working_layer_id, f"MCP Subject: {label}")
                mask = self._apply_vision_mask(
                    image_id,
                    working_layer_id,
                    refined.mask.path,
                    source["width"],
                    source["height"],
                    output_directory,
                )
                quality = self._local_mask_quality(
                    refined.mask.path, source["width"], source["height"]
                )
                if quality["all_transparent"] or quality["all_opaque"]:
                    raise RuntimeError(
                        "semantic vision produced a pathological all-transparent or all-opaque mask"
                    )
                if source_visibility is True:
                    self.set_layer_visibility(layer_id, False)
                self.set_selected_layers(image_id, [working_layer_id])
                return {
                    "status": "completed",
                    "image_id": image_id,
                    "source_layer_id": layer_id,
                    "working_layer_id": working_layer_id,
                    "mask_id": mask["mask_id"],
                    "prompt": prompt.strip(),
                    "strategy": "local-semantic-vision",
                    "provider": result.provider,
                    "model": result.model,
                    "candidate": {
                        "candidate_id": candidate.candidate_id,
                        "concept": candidate.concept,
                        "score": candidate.score,
                        "bounding_box": candidate.bounding_box.as_dict()
                        if candidate.bounding_box
                        else None,
                    },
                    "refinement": refined.strategy,
                    "quality": quality,
                    "quality_is_proxy": True,
                    "runtime_seconds": round(time.perf_counter() - started, 3),
                    "source_preserved": True,
                    "source_visibility_before": source_visibility,
                    "source_visibility_after": False
                    if source_visibility is True
                    else source_visibility,
                    "warnings": list(result.warnings) + list(refined.warnings),
                    "limitations": [
                        "Provider confidence is not semantic segmentation accuracy.",
                        "The current refiner preserves provider alpha but is not learned "
                        "alpha matting.",
                        "No ground-truth fox alpha matte is available for this image.",
                    ],
                }
            except Exception:
                if working_layer_id is not None:
                    try:
                        self.delete_layer(image_id, working_layer_id)
                    except Exception:
                        logger.exception("failed to clean up semantic subject layer")
                raise

    @staticmethod
    def _vision_layer_label(value: str) -> str:
        return " ".join(value.strip().split())[:96]

    def _segment_decomposition_concept(
        self,
        snapshot: Path,
        output_directory: Path,
        concept: str,
        width: int,
        height: int,
        *,
        max_candidates: int,
        minimum_score: float,
    ) -> SegmentationResult:
        request = self._vision_request(
            snapshot,
            output_directory,
            concept,
            max_candidates=max_candidates,
            minimum_score=minimum_score,
            mode="instance",
        )
        return self._validate_vision_result(
            self.vision.segment(request),
            output_directory,
            width,
            height,
            minimum_score,
        )

    @staticmethod
    def _decomposition_candidate_summary(candidate: SegmentationCandidate) -> dict[str, Any]:
        return {
            "candidate_id": candidate.candidate_id,
            "concept": candidate.concept,
            "score": candidate.score,
            "bounding_box": candidate.bounding_box.as_dict() if candidate.bounding_box else None,
            "metadata": dict(candidate.metadata),
        }

    def separate_subject_to_layers(
        self,
        image_id: int,
        layer_id: int,
        prompt: str,
        *,
        include_background: bool = True,
        create_group: bool = True,
        preserve_original: bool = True,
        minimum_score: float = 0.0,
    ) -> dict[str, Any]:
        """Leave one prompted subject and its exact-complement background as GIMP layers."""

        result = self._separate_concepts_to_layers(
            image_id,
            layer_id,
            [prompt],
            include_background=include_background,
            create_group=create_group,
            preserve_original=preserve_original,
            instance_mode="merge",
            overlap_policy="report",
            max_instances_per_concept=1,
            minimum_score=minimum_score,
            subject_style=True,
        )
        result["operation"] = "separate_subject_to_layers"
        result["prompt"] = prompt.strip()
        return result

    def separate_concepts_to_layers(
        self,
        image_id: int,
        layer_id: int,
        concepts: list[str],
        *,
        include_background: bool = True,
        create_group: bool = True,
        preserve_original: bool = True,
        instance_mode: str = "separate",
        overlap_policy: str = "report",
        max_instances_per_concept: int = 4,
        minimum_score: float = 0.0,
    ) -> dict[str, Any]:
        """Create bounded concept/instance layers and a remainder from semantic masks."""

        return self._separate_concepts_to_layers(
            image_id,
            layer_id,
            concepts,
            include_background=include_background,
            create_group=create_group,
            preserve_original=preserve_original,
            instance_mode=instance_mode,
            overlap_policy=overlap_policy,
            max_instances_per_concept=max_instances_per_concept,
            minimum_score=minimum_score,
            subject_style=False,
        )

    def _separate_concepts_to_layers(
        self,
        image_id: int,
        layer_id: int,
        concepts: list[str],
        *,
        include_background: bool,
        create_group: bool,
        preserve_original: bool,
        instance_mode: str,
        overlap_policy: str,
        max_instances_per_concept: int,
        minimum_score: float,
        subject_style: bool,
    ) -> dict[str, Any]:
        self._id(image_id, "image_id")
        self._id(layer_id, "layer_id")
        self._assert_item_belongs_to_image(image_id, layer_id)
        if not isinstance(concepts, list) or not 1 <= len(concepts) <= _MAX_DECOMPOSITION_CONCEPTS:
            raise ValueError(f"concepts must contain 1 to {_MAX_DECOMPOSITION_CONCEPTS} strings")
        normalized = [
            self._vision_layer_label(value) if isinstance(value, str) else "" for value in concepts
        ]
        if any(not value for value in normalized):
            raise ValueError("each concept must be a non-empty string")
        if len({value.casefold() for value in normalized}) != len(normalized):
            raise ValueError("concepts cannot contain duplicates")
        if instance_mode not in {"separate", "merge"}:
            raise ValueError("instance_mode must be separate or merge")
        if overlap_policy != "report":
            raise ValueError("overlap_policy currently supports only report")
        if not isinstance(include_background, bool) or not isinstance(create_group, bool):
            raise ValueError("include_background and create_group must be boolean")
        if preserve_original is not True:
            raise ValueError(
                "preserve_original=false is not supported; the source must remain intact"
            )
        if (
            not isinstance(max_instances_per_concept, int)
            or isinstance(max_instances_per_concept, bool)
            or not 1 <= max_instances_per_concept <= _MAX_INSTANCES_PER_CONCEPT
        ):
            raise ValueError(
                f"max_instances_per_concept must be between 1 and {_MAX_INSTANCES_PER_CONCEPT}"
            )
        maximum_layers = len(normalized) * (
            max_instances_per_concept if instance_mode == "separate" else 1
        ) + int(include_background)
        if maximum_layers > _MAX_GENERATED_VISION_LAYERS:
            raise ValueError(
                f"requested decomposition can create at most {_MAX_GENERATED_VISION_LAYERS} layers"
            )
        source = self.get_layer_info(image_id, layer_id)
        source_mask = self.masks.get(layer_id)
        if source_mask.has_mask:
            raise ValueError("source layer already has a mask; choose an unmasked source layer")
        if self._selection_snapshot(image_id)["active"]:
            raise ValueError("layer decomposition refuses to replace an existing image selection")
        source_visibility = source.get("visible")
        selected_before = [item["layer_id"] for item in self.get_selected_layers(image_id)]
        started = time.perf_counter()
        generated_layer_ids: list[int] = []
        generated: list[dict[str, Any]] = []
        group_id: int | None = None
        source_hidden = False
        with self._vision_snapshot(image_id) as (snapshot, temporary_directory):
            mask_specs: list[dict[str, Any]] = []
            provider_results: list[SegmentationResult] = []
            for concept_index, concept in enumerate(normalized):
                concept_directory = temporary_directory / f"concept-{concept_index}"
                concept_directory.mkdir()
                result = self._segment_decomposition_concept(
                    snapshot,
                    concept_directory,
                    concept,
                    source["width"],
                    source["height"],
                    max_candidates=max_instances_per_concept,
                    minimum_score=minimum_score,
                )
                provider_results.append(result)
                candidates = sorted(
                    result.candidates,
                    key=lambda item: item.score if item.score is not None else -1.0,
                    reverse=True,
                )[:max_instances_per_concept]
                if instance_mode == "merge" and len(candidates) > 1:
                    merged_path = concept_directory / "merged.png"
                    union_mask_png([item.mask.path for item in candidates], merged_path)
                    mask_specs.append(
                        {
                            "concept": concept,
                            "name": concept,
                            "path": merged_path,
                            "candidates": [
                                self._decomposition_candidate_summary(item) for item in candidates
                            ],
                        }
                    )
                else:
                    for instance_index, candidate in enumerate(candidates, start=1):
                        suffix = f" {instance_index}" if len(candidates) > 1 else ""
                        mask_specs.append(
                            {
                                "concept": concept,
                                "name": f"{concept}{suffix}",
                                "path": candidate.mask.path,
                                "candidates": [self._decomposition_candidate_summary(candidate)],
                            }
                        )
            if not mask_specs:
                raise RuntimeError("vision provider returned no usable decomposition masks")
            if len(mask_specs) + int(include_background) > _MAX_GENERATED_VISION_LAYERS:
                raise RuntimeError("provider results exceed the generated-layer safety bound")
            foreground_paths = [spec["path"] for spec in mask_specs]
            overlap = mask_overlap_statistics(foreground_paths)
            union_path = temporary_directory / "foreground-union.png"
            union_mask_png(foreground_paths, union_path)
            background_path = temporary_directory / "background-complement.png"
            if include_background:
                complement_mask_png(union_path, background_path)
            for spec in mask_specs:
                quality = self._local_mask_quality(spec["path"], source["width"], source["height"])
                if quality["all_transparent"] or quality["all_opaque"]:
                    raise RuntimeError(
                        f"semantic vision produced a pathological mask for {spec['concept']}"
                    )
                spec["quality"] = quality
            try:
                if create_group:
                    group_name = (
                        f"MCP Vision — {normalized[0]}"
                        if subject_style
                        else "MCP Vision Decomposition"
                    )
                    group_id = self.create_layer_group(image_id, group_name)["layer_id"]
                for position, spec in enumerate(mask_specs):
                    duplicate = self.duplicate_layer(image_id, layer_id)
                    generated_id = duplicate["layer_id"]
                    generated_layer_ids.append(generated_id)
                    layer_name = f"Subject — {spec['name']}" if subject_style else spec["name"]
                    self.rename_layer(generated_id, layer_name)
                    if group_id is not None:
                        self.move_layer(image_id, generated_id, position, group_id)
                    mask = self._apply_vision_mask(
                        image_id,
                        generated_id,
                        spec["path"],
                        source["width"],
                        source["height"],
                        temporary_directory,
                    )
                    generated.append(
                        {
                            "role": "subject" if subject_style else "concept",
                            "concept": spec["concept"],
                            "layer_id": generated_id,
                            "name": layer_name,
                            "mask_id": mask["mask_id"],
                            "quality": spec["quality"],
                            "candidates": spec["candidates"],
                        }
                    )
                if include_background:
                    background = self.duplicate_layer(image_id, layer_id)
                    background_id = background["layer_id"]
                    generated_layer_ids.append(background_id)
                    background_name = "Background" if subject_style else "Remainder Background"
                    self.rename_layer(background_id, background_name)
                    if group_id is not None:
                        self.move_layer(image_id, background_id, len(mask_specs), group_id)
                    background_mask = self._apply_vision_mask(
                        image_id,
                        background_id,
                        background_path,
                        source["width"],
                        source["height"],
                        temporary_directory,
                    )
                    generated.append(
                        {
                            "role": "background",
                            "concept": None,
                            "layer_id": background_id,
                            "name": background_name,
                            "mask_id": background_mask["mask_id"],
                            "quality": self._local_mask_quality(
                                background_path, source["width"], source["height"]
                            ),
                            "mask_relationship": "exact 8-bit complement of foreground union",
                        }
                    )
                if source_visibility is True:
                    self.set_layer_visibility(layer_id, False)
                    source_hidden = True
                foreground_layer_ids = [
                    item["layer_id"] for item in generated if item["role"] != "background"
                ]
                self.set_selected_layers(image_id, foreground_layer_ids)
            except Exception:
                for generated_id in reversed(generated_layer_ids):
                    try:
                        self.delete_layer(image_id, generated_id)
                    except Exception:
                        logger.exception(
                            "failed to roll back generated vision layer %s", generated_id
                        )
                if group_id is not None:
                    try:
                        self.delete_layer(image_id, group_id)
                    except Exception:
                        logger.exception("failed to roll back generated vision group %s", group_id)
                if source_hidden:
                    try:
                        self.set_layer_visibility(layer_id, bool(source_visibility))
                    except Exception:
                        logger.exception("failed to restore source-layer visibility")
                if selected_before:
                    try:
                        self.set_selected_layers(image_id, selected_before)
                    except Exception:
                        logger.exception("failed to restore selected layers")
                raise
            providers = [
                {
                    "concept": concept,
                    "provider": result.provider,
                    "model": result.model,
                    "runtime_seconds": result.runtime_seconds,
                    "warnings": list(result.warnings),
                    "provenance": dict(result.provenance),
                }
                for concept, result in zip(normalized, provider_results, strict=True)
            ]
            return {
                "status": "completed",
                "operation": "separate_concepts_to_layers",
                "image_id": image_id,
                "source_layer_id": layer_id,
                "source_preserved": True,
                "source_mask_before": source_mask.as_dict(),
                "source_visibility_before": source_visibility,
                "source_visibility_after": False
                if source_visibility is True
                else source_visibility,
                "source_visibility_changed": source_hidden,
                "group_id": group_id,
                "create_group": create_group,
                "include_background": include_background,
                "instance_mode": instance_mode,
                "overlap_policy": overlap_policy,
                "overlap": overlap,
                "background_basis": "exact complement of the soft-alpha foreground union"
                if include_background
                else None,
                "generated_layers": generated,
                "generated_layer_count": len(generated),
                "selected_layer_ids": [
                    item["layer_id"] for item in generated if item["role"] != "background"
                ],
                "providers": providers,
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "quality_is_proxy": True,
                "limitations": [
                    "Provider scores and activation values are not segmentation accuracy.",
                    "Independently prompted masks may overlap; report mode does not assign "
                    "ownership.",
                    "Semantic probability masks are not learned alpha mattes.",
                    "GIMP may expose partial mask samples in image-space encoding even when the "
                    "provider artifacts are exact complements.",
                ],
            }

    def isolate_subject(
        self,
        image_id: int,
        layer_id: int,
        strategy: str = "auto",
        background_threshold: int = 48,
        refinement_threshold: int = 24,
        feather: int = 1,
        prompt: str = "subject",
    ) -> dict[str, Any]:
        """Select semantic vision when available, otherwise use the explicit heuristic fallback."""

        if strategy not in {"auto", "high-key-background", "border-color", "vision"}:
            raise ValueError("strategy must be auto, vision, high-key-background, or border-color")
        if strategy == "vision":
            return self.isolate_subject_vision(image_id, layer_id, prompt)
        vision = getattr(self, "vision", None)
        if strategy == "auto" and vision is not None:
            capabilities = vision.capabilities()
            if capabilities.available and capabilities.text_segmentation:
                return self.isolate_subject_vision(image_id, layer_id, prompt)
        return self._isolate_subject_heuristic(
            image_id,
            layer_id,
            strategy,
            background_threshold,
            refinement_threshold,
            feather,
        )

    def _isolate_subject_heuristic(
        self,
        image_id: int,
        layer_id: int,
        strategy: str = "auto",
        background_threshold: int = 48,
        refinement_threshold: int = 24,
        feather: int = 1,
    ) -> dict[str, Any]:
        """Duplicate a layer and create a bounded, non-destructive subject mask.

        The current strategy is deliberately limited to high-key backgrounds. It seeds
        contiguous-color selection from sufficiently light perimeter pixels, then converts
        the inverted selection into a layer mask. The refined pass uses a lower threshold
        and modest feathering to keep uncertain fur transitions instead of hard-binarizing.
        """

        self._id(image_id, "image_id")
        self._id(layer_id, "layer_id")
        self._assert_item_belongs_to_image(image_id, layer_id)
        if strategy not in {"auto", "high-key-background", "border-color"}:
            raise ValueError("strategy must be auto, high-key-background, or border-color")
        for value, name in (
            (background_threshold, "background_threshold"),
            (refinement_threshold, "refinement_threshold"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 255:
                raise ValueError(f"{name} must be an integer from 1 to 255")
        if refinement_threshold > background_threshold:
            raise ValueError("refinement_threshold cannot exceed background_threshold")
        if not isinstance(feather, int) or isinstance(feather, bool) or not 0 <= feather <= 3:
            raise ValueError("feather must be an integer from 0 to 3")
        source = self.get_layer_info(image_id, layer_id)
        existing_mask = self.masks.get(layer_id)
        if existing_mask.has_mask:
            raise ValueError("source layer already has a mask; choose an unmasked source layer")
        selection_before = self._selection_snapshot(image_id)
        if selection_before["active"]:
            raise ValueError("isolate_subject refuses to replace an existing image selection")
        old_threshold = _scalar(self.evaluate(scheme_call("gimp-context-get-sample-threshold-int")))
        if not isinstance(old_threshold, int):
            raise RuntimeError(f"GIMP returned malformed sample threshold: {old_threshold!r}")
        border = self._border_characteristics(layer_id, source["width"], source["height"])
        use_all_border = border["high_key_ratio"] >= 0.6
        border["selection_policy"] = (
            "all-perimeter-samples" if use_all_border else "high-key-perimeter-samples"
        )
        started = time.perf_counter()
        baseline_layer_id: int | None = None
        final_layer_id: int | None = None
        final_mask_id: int | None = None
        source_visibility = source.get("visible")
        try:
            baseline = self.duplicate_layer(image_id, layer_id)
            baseline_layer_id = baseline["layer_id"]
            self.rename_layer(baseline_layer_id, "MCP Subject Isolation (baseline)")
            self._select_border_background(
                image_id,
                baseline_layer_id,
                source["width"],
                source["height"],
                threshold=background_threshold,
                feather=0,
                minimum_luminance=None if use_all_border else 170,
            )
            baseline_mask = self.masks.create_and_attach(baseline_layer_id, "selection")
            if baseline_mask.mask_id is None:
                raise RuntimeError("baseline mask was not assigned an identity")
            baseline_samples = self._sample_mask(
                baseline_mask.mask_id, source["width"], source["height"]
            )
            baseline_quality = mask_metrics(baseline_samples, source["width"], source["height"])
            self.delete_layer(image_id, baseline_layer_id)
            baseline_layer_id = None

            final = self.duplicate_layer(image_id, layer_id)
            final_layer_id = final["layer_id"]
            self.rename_layer(final_layer_id, "MCP Subject Isolation")
            self._select_border_background(
                image_id,
                final_layer_id,
                source["width"],
                source["height"],
                threshold=refinement_threshold,
                feather=feather,
                minimum_luminance=None if use_all_border else 170,
            )
            final_mask = self.masks.create_and_attach(final_layer_id, "selection")
            final_mask_id = final_mask.mask_id
            if final_mask_id is None:
                raise RuntimeError("final mask was not assigned an identity")
            final_samples = self._sample_mask(final_mask_id, source["width"], source["height"])
            final_quality = mask_metrics(
                final_samples,
                source["width"],
                source["height"],
                foreground_reference=[alpha for _, _, alpha in baseline_samples],
            )
            if final_quality["all_transparent"] or final_quality["all_opaque"]:
                raise RuntimeError(
                    "subject isolation produced a pathological all-transparent or all-opaque mask"
                )
            if source_visibility is True:
                self.set_layer_visibility(layer_id, False)
            self.set_selected_layers(image_id, [final_layer_id])
            return {
                "status": "completed",
                "image_id": image_id,
                "source_layer_id": layer_id,
                "working_layer_id": final_layer_id,
                "mask_id": final_mask_id,
                "strategy": "high-key-border-contiguous-color",
                "parameters": {
                    "background_threshold": background_threshold,
                    "refinement_threshold": refinement_threshold,
                    "feather": feather,
                },
                "border_evidence": border,
                "baseline": baseline_quality,
                "final": final_quality,
                "quality_is_proxy": True,
                "runtime_seconds": round(time.perf_counter() - started, 3),
                "source_preserved": True,
                "source_visibility_before": source_visibility,
                "source_visibility_after": False
                if source_visibility is True
                else source_visibility,
                "limitations": [
                    "Native foreground extraction was unavailable through Script-Fu on GIMP 3.2.0.",
                    "Metrics are bounded mask observables, not ground-truth segmentation accuracy.",
                    "Border-seeded high-key isolation is not suitable for every background.",
                ],
            }
        except Exception:
            if final_layer_id is not None:
                try:
                    self.delete_layer(image_id, final_layer_id)
                except Exception:
                    logger.exception("failed to clean up subject-isolation working layer")
            raise
        finally:
            if baseline_layer_id is not None:
                try:
                    self.delete_layer(image_id, baseline_layer_id)
                except Exception:
                    logger.exception("failed to clean up subject-isolation baseline layer")
            try:
                self.evaluate(scheme_call("gimp-selection-none", [image_id]))
            finally:
                self.evaluate(scheme_call("gimp-context-set-sample-threshold-int", [old_threshold]))

    def brightness_contrast(
        self, layer_id: int, brightness: float, contrast: float
    ) -> dict[str, Any]:
        if not -1 <= brightness <= 1 or not -1 <= contrast <= 1:
            raise ValueError("Brightness and contrast must be between -1 and 1")
        self.evaluate(
            scheme_call("gimp-drawable-brightness-contrast", [layer_id, brightness, contrast])
        )
        return {"layer_id": layer_id, "brightness": brightness, "contrast": contrast}

    def apply_gaussian_blur_filter(
        self,
        layer_id: int,
        radius_x: float,
        radius_y: float | None = None,
        *,
        name: str = "Gaussian Blur",
        opacity: float = 100.0,
        blend_mode: str = "LAYER-MODE-REPLACE",
    ) -> dict[str, Any]:
        self._id(layer_id, "layer_id")
        radius_y = radius_x if radius_y is None else radius_y
        self._finite_range(radius_x, 0, 1000, "radius_x")
        self._finite_range(radius_y, 0, 1000, "radius_y")
        spec = DrawableFilterSpec.create(
            layer_id,
            "gegl:gaussian-blur",
            name,
            blend_mode=blend_mode,
            opacity=opacity,
            parameters={"std-dev-x": radius_x, "std-dev-y": radius_y},
        )
        return self.filters.append(spec).as_dict()

    def apply_brightness_contrast_filter(
        self,
        layer_id: int,
        brightness: float,
        contrast: float,
        *,
        name: str = "Brightness / Contrast",
        opacity: float = 100.0,
        blend_mode: str = "LAYER-MODE-REPLACE",
    ) -> dict[str, Any]:
        self._id(layer_id, "layer_id")
        self._finite_range(brightness, -1, 1, "brightness")
        self._finite_range(contrast, -1, 1, "contrast")
        spec = DrawableFilterSpec.create(
            layer_id,
            "gegl:brightness-contrast",
            name,
            blend_mode=blend_mode,
            opacity=opacity,
            parameters={"brightness": brightness, "contrast": contrast},
        )
        return self.filters.append(spec).as_dict()

    def list_drawable_filters(self, layer_id: int) -> list[dict[str, Any]]:
        self._id(layer_id, "layer_id")
        return [item.as_dict() for item in self.filters.list(layer_id)]

    def hue_saturation(
        self, layer_id: int, hue: float, saturation: float, lightness: float
    ) -> dict[str, Any]:
        if not -180 <= hue <= 180 or not -100 <= saturation <= 100 or not -100 <= lightness <= 100:
            raise ValueError("Hue must be -180..180; saturation and lightness must be -100..100")
        self.evaluate(
            scheme_call("gimp-drawable-hue-saturation", [layer_id, hue, saturation, lightness])
        )
        return {"layer_id": layer_id, "hue": hue, "saturation": saturation, "lightness": lightness}

    def desaturate(self, layer_id: int, mode: str = "DESATURATE-LUMA") -> dict[str, Any]:
        self.evaluate(
            scheme_call("gimp-drawable-desaturate", [layer_id, SchemeSymbol(mode.upper())])
        )
        return {"layer_id": layer_id, "mode": mode.upper()}

    def undo(self, image_id: int) -> dict[str, Any]:
        self.evaluate(scheme_call("gimp-image-undo", [image_id]))
        return {"image_id": image_id, "action": "undo"}

    def redo(self, image_id: int) -> dict[str, Any]:
        self.evaluate(scheme_call("gimp-image-redo", [image_id]))
        return {"image_id": image_id, "action": "redo"}

    def _save(self, image_id: int, path: Path) -> None:
        self._id(image_id, "image_id")
        self.evaluate(
            scheme_call(
                "gimp-file-save",
                [SchemeSymbol("RUN-NONINTERACTIVE"), image_id, str(path), SchemeNull()],
            )
        )

    def _export(self, image_id: int, path: Path) -> None:
        self._id(image_id, "image_id")
        raise ScriptFuError(
            "GIMP 3 Script-Fu does not expose a safe gimp-file-export binding in this "
            "environment; refusing to route export through gimp-file-save"
        )

    def _run_undo_group(self, image_id: int, operation: str) -> Any:
        """Run a generated multi-call operation as one GIMP undo step."""

        expression = (
            f"(let ((result (begin (gimp-image-undo-group-start {image_id}) "
            f"{operation}))) (gimp-image-undo-group-end {image_id}) result)"
        )
        return self.evaluate(expression)

    def _item_position(self, image_id: int, layer_id: int) -> int:
        value = _scalar(
            self.evaluate(scheme_call("gimp-image-get-item-position", [image_id, layer_id]))
        )
        return int(value) if isinstance(value, int) else -1

    def _assert_item_belongs_to_image(self, image_id: int, item_id: int) -> None:
        owner = _scalar(self.evaluate(scheme_call("gimp-item-get-image", [item_id])))
        if not isinstance(owner, int) or owner != image_id:
            raise ValueError(f"Layer {item_id} does not belong to image {image_id}")

    def _validate_parent(self, image_id: int, parent_id: int | None) -> None:
        if parent_id is None:
            return
        self._id(parent_id, "parent_id")
        self._assert_item_belongs_to_image(image_id, parent_id)
        is_group = _scalar(self.evaluate(scheme_call("gimp-item-is-group", [parent_id])))
        if is_group is not True:
            raise ValueError(f"parent_id {parent_id} is not a GIMP layer group")

    @staticmethod
    def _file_value(value: Any) -> str | None:
        if isinstance(value, str) and not value.startswith("#<"):
            return value
        return None

    @staticmethod
    def _scalar_bool(value: Any) -> bool:
        value = _scalar(value)
        if not isinstance(value, bool):
            raise RuntimeError(f"GIMP returned a non-boolean dirty state: {value!r}")
        return value

    @staticmethod
    def _id(value: int, name: str) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative GIMP object ID")

    @staticmethod
    def _size(width: int, height: int) -> None:
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers")

    @staticmethod
    def _finite_range(value: float, minimum: float, maximum: float, name: str) -> None:
        import math

        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not minimum <= float(value) <= maximum
        ):
            raise ValueError(f"{name} must be a finite number from {minimum} to {maximum}")

    @staticmethod
    def _existing_file(path: str) -> Path:
        if not path:
            raise PathPolicyError("An explicit input path is required")
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_file():
            raise PathPolicyError(f"Input file does not exist or is not a file: {candidate}")
        return candidate

    @staticmethod
    def _output_file(path: str, overwrite: bool) -> Path:
        if not path:
            raise PathPolicyError("An explicit output path is required")
        candidate = Path(path).expanduser().resolve()
        if candidate.exists() and candidate.is_dir():
            raise PathPolicyError(f"Output path is a directory: {candidate}")
        if not candidate.parent.is_dir():
            raise PathPolicyError(f"Output directory does not exist: {candidate.parent}")
        if candidate.exists() and not overwrite:
            raise PathPolicyError(
                f"Refusing to overwrite existing file without overwrite=true: {candidate}"
            )
        return candidate
