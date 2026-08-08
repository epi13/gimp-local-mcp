"""Stable, JSON-compatible MCP-facing state models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ImageInfo:
    image_id: int
    name: str | None
    width: int
    height: int
    base_type: str | int | None
    file: str | None
    dirty: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LayerInfo:
    layer_id: int
    image_id: int
    name: str | None
    width: int
    height: int
    visible: bool | None
    opacity: float | int | None
    mode: str | int | None
    position: int
    parent_id: int | None
    is_group: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LayerNode:
    """A bounded recursive layer-tree node reported by GIMP."""

    info: LayerInfo
    children: tuple[LayerNode, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        result = self.info.as_dict()
        result["children"] = [child.as_dict() for child in self.children]
        return result
