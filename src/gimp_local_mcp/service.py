"""High-level GIMP operations composed over the transport and PDB gateway."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import Config
from .errors import PathPolicyError, UnsafeOperationError
from .gimp.pdb import PdbCatalog, PdbInvoker
from .gimp.scheme import parse_scheme, unwrap
from .gimp.serializer import SchemeNull, SchemeSymbol, scheme_call, with_v3
from .gimp.transport import ScriptFuClient
from .models import ImageInfo, LayerInfo

logger = logging.getLogger(__name__)


def _scalar(value: Any) -> Any:
    value = unwrap(value)
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _object_ids(value: Any) -> list[int]:
    value = unwrap(value)
    if isinstance(value, list):
        if len(value) >= 2 and isinstance(value[1], list):
            return [int(item) for item in value[1] if isinstance(item, int) and item >= 0]
        if len(value) == 1 and isinstance(value[0], list):
            return [int(item) for item in value[0] if isinstance(item, int) and item >= 0]
        if all(isinstance(item, int) for item in value):
            return [int(item) for item in value if item >= 0]
    return []


class GimpService:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.from_env()
        self.client = ScriptFuClient(self.config)
        self.catalog = PdbCatalog(self.client)
        self.pdb = PdbInvoker(self.client, self.catalog)

    def close(self) -> None:
        self.client.close()

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
            "pdb_discovery": ["search", "exists", "procedure counts", "documentation"],
            "high_level_tools": [
                "images",
                "layers",
                "transforms",
                "selections",
                "undo",
                "basic adjustments",
            ],
            "limitations": [
                "Active image selection depends on GIMP's default display.",
                "PDB argument GParamSpec names are not yet exposed by Script-Fu.",
                "Export metadata options are left to GIMP's configured defaults in v0.1.",
            ],
        }

    def list_open_images(self) -> list[dict[str, Any]]:
        result = self.evaluate(scheme_call("gimp-get-images"))
        return [self.get_image_info(image_id) for image_id in _object_ids(result)]

    def get_active_image(self) -> dict[str, Any] | None:
        display = _scalar(self.evaluate(scheme_call("gimp-default-display")))
        if not isinstance(display, int) or display < 0:
            return None
        image_id = _scalar(self.evaluate(scheme_call("gimp-display-get-image", [display])))
        if not isinstance(image_id, int) or image_id < 0:
            return None
        return self.get_image_info(image_id)

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
        self._save(image_id, target)
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
        ids = _object_ids(self.evaluate(scheme_call("gimp-image-get-layers", [image_id])))
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
                f"(gimp-item-get-parent {layer_id}))"
            )
        )
        values = values if isinstance(values, list) else []
        parent = (
            values[6] if len(values) > 6 and isinstance(values[6], int) and values[6] >= 0 else None
        )
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
        )
        return info.as_dict()

    def create_layer(
        self,
        image_id: int,
        name: str,
        width: int | None = None,
        height: int | None = None,
        opacity: float = 100.0,
    ) -> dict[str, Any]:
        self._id(image_id, "image_id")
        if not name:
            raise ValueError("Layer name cannot be empty")
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
            f"(let ((layer {layer_expr})) (gimp-image-insert-layer {image_id} layer -1 0) layer)"
        )
        layer_id = _scalar(self._run_undo_group(image_id, expression))
        if not isinstance(layer_id, int) or layer_id < 0:
            raise RuntimeError("GIMP did not return a layer ID when creating a layer")
        return self.get_layer_info(image_id, layer_id, 0)

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
        self.evaluate(
            scheme_call(
                "gimp-image-select-item", [image_id, SchemeSymbol("CHANNEL-OP-REPLACE"), layer_id]
            )
        )
        return {"image_id": image_id, "layer_id": layer_id, "selection": "layer-alpha"}

    def brightness_contrast(
        self, layer_id: int, brightness: float, contrast: float
    ) -> dict[str, Any]:
        if not -1 <= brightness <= 1 or not -1 <= contrast <= 1:
            raise ValueError("Brightness and contrast must be between -1 and 1")
        self.evaluate(
            scheme_call("gimp-drawable-brightness-contrast", [layer_id, brightness, contrast])
        )
        return {"layer_id": layer_id, "brightness": brightness, "contrast": contrast}

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
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative GIMP object ID")

    @staticmethod
    def _size(width: int, height: int) -> None:
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers")

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
