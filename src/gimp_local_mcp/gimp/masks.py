"""Safe, structured access to GIMP layer masks."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .scheme import unwrap
from .serializer import SchemeSymbol, scheme_call

_MASK_TYPES = {
    "white": "ADD-MASK-WHITE",
    "black": "ADD-MASK-BLACK",
    "alpha": "ADD-MASK-ALPHA",
    "selection": "ADD-MASK-SELECTION",
    "copy": "ADD-MASK-COPY",
}
_REMOVE_MASK_TYPES = {"discard": "MASK-DISCARD", "apply": "MASK-APPLY"}


def _scalar(value: Any) -> Any:
    value = unwrap(value)
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


@dataclass(frozen=True, slots=True)
class LayerMaskInfo:
    layer_id: int
    mask_id: int | None
    has_mask: bool
    enabled: bool | None
    visible: bool | None
    editable: bool | None
    width: int | None
    height: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "mask_id": self.mask_id,
            "has_mask": self.has_mask,
            "enabled": self.enabled,
            "visible": self.visible,
            "editable": self.editable,
            "width": self.width,
            "height": self.height,
        }


class LayerMaskGateway:
    """Execute bounded mask operations and read state reported by GIMP."""

    def __init__(self, evaluate: Callable[[str], Any]) -> None:
        self._evaluate = evaluate

    @staticmethod
    def _id(value: int, label: str) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{label} must be a non-negative GIMP object ID")

    def get(self, layer_id: int) -> LayerMaskInfo:
        self._id(layer_id, "layer_id")
        mask_id = _scalar(self._evaluate(scheme_call("gimp-layer-get-mask", [layer_id])))
        if mask_id in {-1, None}:
            return LayerMaskInfo(layer_id, None, False, None, None, None, None, None)
        self._id(mask_id, "mask_id")

        def read(name: str) -> Any:
            return _scalar(self._evaluate(scheme_call(name, [layer_id])))

        enabled, visible, editable = (
            read("gimp-layer-get-apply-mask"),
            read("gimp-layer-get-show-mask"),
            read("gimp-layer-get-edit-mask"),
        )
        for value, label in (
            (enabled, "enabled"),
            (visible, "visible"),
            (editable, "editable"),
        ):
            if not isinstance(value, bool):
                raise RuntimeError(f"GIMP returned malformed mask {label}: {value!r}")
        width = _scalar(self._evaluate(scheme_call("gimp-drawable-get-width", [mask_id])))
        height = _scalar(self._evaluate(scheme_call("gimp-drawable-get-height", [mask_id])))
        if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
            raise RuntimeError(f"GIMP returned malformed mask dimensions: {width!r}x{height!r}")
        return LayerMaskInfo(layer_id, mask_id, True, enabled, visible, editable, width, height)

    def create(self, layer_id: int, mask_type: str = "selection") -> LayerMaskInfo:
        self._id(layer_id, "layer_id")
        normalized = mask_type.strip().lower() if isinstance(mask_type, str) else ""
        if normalized not in _MASK_TYPES:
            raise ValueError(f"mask_type must be one of: {', '.join(sorted(_MASK_TYPES))}")
        existing = self.get(layer_id)
        if existing.has_mask:
            raise ValueError(f"layer {layer_id} already has a mask; refusing to replace it")
        mask_id = _scalar(
            self._evaluate(
                scheme_call(
                    "gimp-layer-create-mask", [layer_id, SchemeSymbol(_MASK_TYPES[normalized])]
                )
            )
        )
        self._id(mask_id, "mask_id")
        return LayerMaskInfo(layer_id, mask_id, False, None, None, None, None, None)

    def attach(self, layer_id: int, mask_id: int) -> LayerMaskInfo:
        self._id(layer_id, "layer_id")
        self._id(mask_id, "mask_id")
        existing = self.get(layer_id)
        if existing.has_mask:
            raise ValueError(f"layer {layer_id} already has a mask; refusing to replace it")
        self._evaluate(scheme_call("gimp-layer-add-mask", [layer_id, mask_id]))
        info = self.get(layer_id)
        if info.mask_id != mask_id:
            raise RuntimeError(f"GIMP attached mask read back as {info.mask_id!r}, not {mask_id}")
        return info

    def create_and_attach(self, layer_id: int, mask_type: str = "selection") -> LayerMaskInfo:
        created = self.create(layer_id, mask_type)
        if created.mask_id is None:
            raise RuntimeError("GIMP returned no mask identity")
        return self.attach(layer_id, created.mask_id)

    def set_enabled(self, layer_id: int, enabled: bool) -> LayerMaskInfo:
        self._id(layer_id, "layer_id")
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be boolean")
        current = self.get(layer_id)
        if not current.has_mask:
            raise ValueError(f"layer {layer_id} has no mask")
        self._evaluate(scheme_call("gimp-layer-set-apply-mask", [layer_id, enabled]))
        result = self.get(layer_id)
        if result.enabled is not enabled:
            raise RuntimeError(f"GIMP mask enabled state read back as {result.enabled!r}")
        return result

    def set_visible(self, layer_id: int, visible: bool) -> LayerMaskInfo:
        self._id(layer_id, "layer_id")
        if not isinstance(visible, bool):
            raise ValueError("visible must be boolean")
        if not self.get(layer_id).has_mask:
            raise ValueError(f"layer {layer_id} has no mask")
        self._evaluate(scheme_call("gimp-layer-set-show-mask", [layer_id, visible]))
        return self.get(layer_id)

    def remove(self, layer_id: int, disposition: str = "discard") -> LayerMaskInfo:
        self._id(layer_id, "layer_id")
        normalized = disposition.lower() if isinstance(disposition, str) else ""
        if normalized not in _REMOVE_MASK_TYPES:
            raise ValueError("disposition must be discard or apply")
        if not self.get(layer_id).has_mask:
            return self.get(layer_id)
        self._evaluate(
            scheme_call(
                "gimp-layer-remove-mask",
                [layer_id, SchemeSymbol(_REMOVE_MASK_TYPES[normalized])],
            )
        )
        return self.get(layer_id)

    @staticmethod
    def validate_alpha(value: Any) -> int:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise RuntimeError(f"GIMP returned malformed mask pixel: {value!r}")
        if not 0 <= float(value) <= 255:
            raise RuntimeError(f"GIMP returned out-of-range mask pixel: {value!r}")
        return int(round(float(value)))
