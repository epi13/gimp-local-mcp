"""Safe, structured access to GIMP's non-destructive drawable filters."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .scheme import unwrap
from .serializer import SchemeSymbol, scheme_call

_OPERATION_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,255}$")
_BLEND_MODE_RE = re.compile(r"^LAYER-MODE-[A-Z0-9_-]{1,80}$")
_MAX_FILTER_PARAMETERS = 32
_MAX_FILTER_NAME = 256


@dataclass(frozen=True, slots=True)
class DrawableFilterSpec:
    """Validated inputs for one non-destructive drawable filter operation."""

    drawable_id: int
    operation: str
    name: str
    blend_mode: str
    opacity: float
    parameters: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def create(
        cls,
        drawable_id: int,
        operation: str,
        name: str,
        *,
        blend_mode: str = "LAYER-MODE-REPLACE",
        opacity: float = 100.0,
        parameters: Mapping[str, Any] | None = None,
    ) -> DrawableFilterSpec:
        if not isinstance(drawable_id, int) or drawable_id < 0:
            raise ValueError("drawable_id must be a non-negative GIMP object ID")
        if not isinstance(operation, str) or not _OPERATION_RE.fullmatch(operation):
            raise ValueError("operation must be a bounded GEGL operation name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("filter name cannot be empty")
        if len(name) > _MAX_FILTER_NAME:
            raise ValueError(f"filter name cannot exceed {_MAX_FILTER_NAME} characters")
        if not isinstance(blend_mode, str) or not _BLEND_MODE_RE.fullmatch(blend_mode):
            raise ValueError("blend_mode must be a GIMP LAYER-MODE enum name")
        if not isinstance(opacity, (int, float)) or isinstance(opacity, bool):
            raise ValueError("filter opacity must be a finite number from 0 to 100")
        if not math.isfinite(float(opacity)) or not 0 <= float(opacity) <= 100:
            raise ValueError("filter opacity must be a finite number from 0 to 100")
        items = tuple((key, value) for key, value in (parameters or {}).items())
        if len(items) > _MAX_FILTER_PARAMETERS:
            raise ValueError(f"a filter can have at most {_MAX_FILTER_PARAMETERS} parameters")
        # scheme_call validates parameter names and every nested value without accepting source.
        spec = cls(
            drawable_id=drawable_id,
            operation=operation,
            name=name,
            blend_mode=blend_mode,
            opacity=float(opacity),
            parameters=items,
        )
        spec.expression()
        return spec

    def expression(self) -> str:
        """Build the GIMP 3 Script-Fu filter call without raw Scheme inputs."""

        return scheme_call(
            "gimp-drawable-append-new-filter",
            [
                self.drawable_id,
                self.operation,
                self.name,
                # GIMP's Script-Fu binding expects opacity in the 0.0..1.0 range.
                SchemeSymbol(self.blend_mode),
                self.opacity / 100.0,
            ],
            keywords=dict(self.parameters),
        )


@dataclass(frozen=True, slots=True)
class DrawableFilterInfo:
    """State reported by GIMP for an attached drawable filter."""

    filter_id: int
    drawable_id: int
    name: str | None
    operation: str | None
    opacity: float | None
    blend_mode: int | str | None
    visible: bool | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "filter_id": self.filter_id,
            "drawable_id": self.drawable_id,
            "name": self.name,
            "operation": self.operation,
            "opacity": self.opacity,
            "blend_mode": self.blend_mode,
            "visible": self.visible,
            "non_destructive": True,
        }


def _scalar(value: Any) -> Any:
    value = unwrap(value)
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


class DrawableFilterGateway:
    """Execute validated filter specs and read only GIMP-reported filter state."""

    def __init__(self, evaluate: Callable[[str], Any]) -> None:
        self._evaluate = evaluate

    def append(self, spec: DrawableFilterSpec) -> DrawableFilterInfo:
        result = _scalar(self._evaluate(spec.expression()))
        if not isinstance(result, int) or isinstance(result, bool) or result < 0:
            raise RuntimeError(f"GIMP returned an invalid drawable filter ID: {result!r}")
        return self.get(spec.drawable_id, result)

    def list(self, drawable_id: int) -> list[DrawableFilterInfo]:
        if not isinstance(drawable_id, int) or drawable_id < 0:
            raise ValueError("drawable_id must be a non-negative GIMP object ID")
        values = unwrap(self._evaluate(scheme_call("gimp-drawable-get-filters", [drawable_id])))
        if not isinstance(values, list):
            raise RuntimeError(f"GIMP returned an invalid drawable filter list: {values!r}")
        filter_ids = [value for value in values if isinstance(value, int) and value >= 0]
        if len(filter_ids) != len(values):
            raise RuntimeError(f"GIMP returned malformed drawable filter IDs: {values!r}")
        return [self.get(drawable_id, filter_id) for filter_id in filter_ids]

    def get(self, drawable_id: int, filter_id: int) -> DrawableFilterInfo:
        if not isinstance(drawable_id, int) or drawable_id < 0:
            raise ValueError("drawable_id must be a non-negative GIMP object ID")
        if not isinstance(filter_id, int) or isinstance(filter_id, bool) or filter_id < 0:
            raise ValueError("filter_id must be a non-negative GIMP object ID")

        def read(name: str) -> Any:
            return _scalar(self._evaluate(scheme_call(name, [filter_id])))

        name = read("gimp-drawable-filter-get-name")
        operation = read("gimp-drawable-filter-get-operation-name")
        opacity = read("gimp-drawable-filter-get-opacity")
        blend_mode = read("gimp-drawable-filter-get-blend-mode")
        visible = read("gimp-drawable-filter-get-visible")
        if name is not None and not isinstance(name, str):
            raise RuntimeError(f"GIMP returned malformed filter name: {name!r}")
        if operation is not None and not isinstance(operation, str):
            raise RuntimeError(f"GIMP returned malformed filter operation: {operation!r}")
        if opacity is not None and (
            not isinstance(opacity, (int, float)) or not math.isfinite(float(opacity))
        ):
            raise RuntimeError(f"GIMP returned malformed filter opacity: {opacity!r}")
        if visible is not None and not isinstance(visible, bool):
            raise RuntimeError(f"GIMP returned malformed filter visibility: {visible!r}")
        return DrawableFilterInfo(
            filter_id=filter_id,
            drawable_id=drawable_id,
            name=name,
            operation=operation,
            opacity=float(opacity) if opacity is not None else None,
            blend_mode=blend_mode,
            visible=visible,
        )
