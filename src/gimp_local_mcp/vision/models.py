"""Bounded, provider-neutral semantic segmentation models."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_MAX_PROMPT = 256
_MAX_CANDIDATES = 32
_MAX_WARNINGS = 64


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bounded_text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > limit:
        raise ValueError(f"{name} exceeds the {limit}-character limit")
    return value.strip()


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """A pixel-space candidate box; it is metadata, not a mask substitute."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        for name, value in (
            ("x", self.x),
            ("y", self.y),
            ("width", self.width),
            ("height", self.height),
        ):
            number = _finite(value, name)
            if number < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("bounding-box width and height must be positive")

    def as_dict(self) -> dict[str, float]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "width": float(self.width),
            "height": float(self.height),
        }

    @classmethod
    def from_dict(cls, value: Any) -> BoundingBox:
        if not isinstance(value, dict):
            raise ValueError("bounding_box must be an object")
        return cls(value.get("x"), value.get("y"), value.get("width"), value.get("height"))


@dataclass(frozen=True, slots=True)
class MaskArtifact:
    """A temporary lossless 8-bit grayscale mask artifact."""

    path: Path
    width: int
    height: int
    soft_alpha: bool = True
    encoding: str = "png"
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("mask artifact path must be an absolute local path")
        if (
            not isinstance(self.width, int)
            or isinstance(self.width, bool)
            or not 1 <= self.width <= 16_384
        ):
            raise ValueError("mask width must be an integer from 1 to 16384")
        if (
            not isinstance(self.height, int)
            or isinstance(self.height, bool)
            or not 1 <= self.height <= 16_384
        ):
            raise ValueError("mask height must be an integer from 1 to 16384")
        if self.encoding != "png":
            raise ValueError("mask artifacts must use lossless PNG encoding")
        if self.sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("mask sha256 must be a lowercase hexadecimal digest")

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "soft_alpha": self.soft_alpha,
            "encoding": self.encoding,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> MaskArtifact:
        if not isinstance(value, dict):
            raise ValueError("mask must be an object")
        path = value.get("path")
        if not isinstance(path, str):
            raise ValueError("mask path must be a string")
        soft_alpha = value.get("soft_alpha", True)
        if not isinstance(soft_alpha, bool):
            raise ValueError("mask soft_alpha must be boolean")
        return cls(
            Path(path),
            value.get("width"),
            value.get("height"),
            soft_alpha,
            value.get("encoding", "png"),
            value.get("sha256"),
        )


@dataclass(frozen=True, slots=True)
class VisionCapabilities:
    provider: str
    available: bool
    model: str | None = None
    text_segmentation: bool = False
    visual_prompts: bool = False
    soft_alpha: bool = False
    backend: str | None = None
    reason: str | None = None
    runtime: str | None = None

    def __post_init__(self) -> None:
        _bounded_text(self.provider, "provider", 64)
        if self.model is not None:
            _bounded_text(self.model, "model", 128)
        if self.reason is not None:
            _bounded_text(self.reason, "reason", 1024)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "available": self.available,
            "model": self.model,
            "text_segmentation": self.text_segmentation,
            "visual_prompts": self.visual_prompts,
            "soft_alpha": self.soft_alpha,
            "backend": self.backend,
            "reason": self.reason,
            "runtime": self.runtime,
        }

    @classmethod
    def from_dict(cls, value: Any) -> VisionCapabilities:
        if not isinstance(value, dict):
            raise ValueError("capabilities must be an object")
        for field_name in ("available", "text_segmentation", "visual_prompts", "soft_alpha"):
            if field_name in value and not isinstance(value[field_name], bool):
                raise ValueError(f"capability {field_name} must be boolean")
        return cls(
            value.get("provider", "unknown"),
            value.get("available") is True,
            value.get("model"),
            value.get("text_segmentation") is True,
            value.get("visual_prompts") is True,
            value.get("soft_alpha") is True,
            value.get("backend"),
            value.get("reason"),
            value.get("runtime"),
        )


@dataclass(frozen=True, slots=True)
class SegmentationRequest:
    image_path: Path
    prompt: str | None = None
    point_prompts: tuple[tuple[float, float, bool], ...] = ()
    box_prompts: tuple[BoundingBox, ...] = ()
    max_candidates: int = 3
    minimum_score: float = 0.0
    mode: str = "semantic"
    output_directory: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.image_path, Path) or not self.image_path.is_absolute():
            raise ValueError("image_path must be an absolute local path")
        if self.prompt is not None:
            _bounded_text(self.prompt, "prompt", _MAX_PROMPT)
        if not 1 <= self.max_candidates <= _MAX_CANDIDATES:
            raise ValueError("max_candidates must be between 1 and 32")
        score = _finite(self.minimum_score, "minimum_score")
        if not 0 <= score <= 1:
            raise ValueError("minimum_score must be between 0 and 1")
        if self.mode not in {"semantic", "instance", "automatic"}:
            raise ValueError("mode must be semantic, instance, or automatic")
        if self.output_directory is not None and not self.output_directory.is_absolute():
            raise ValueError("output_directory must be an absolute local path")
        for point in self.point_prompts:
            if not isinstance(point, tuple) or len(point) != 3:
                raise ValueError("point prompts must be (x, y, positive) tuples")
            _finite(point[0], "point x")
            _finite(point[1], "point y")
            if not isinstance(point[2], bool):
                raise ValueError("point prompt polarity must be boolean")

    def as_dict(self, *, include_output_directory: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "image_path": str(self.image_path),
            "prompt": self.prompt,
            "point_prompts": [list(point) for point in self.point_prompts],
            "box_prompts": [box.as_dict() for box in self.box_prompts],
            "max_candidates": self.max_candidates,
            "minimum_score": self.minimum_score,
            "mode": self.mode,
        }
        if include_output_directory:
            result["output_directory"] = (
                str(self.output_directory) if self.output_directory else None
            )
        return result


@dataclass(frozen=True, slots=True)
class SegmentationCandidate:
    candidate_id: str
    concept: str | None
    score: float | None
    bounding_box: BoundingBox | None
    mask: MaskArtifact
    width: int
    height: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _ID_RE.fullmatch(self.candidate_id):
            raise ValueError("candidate_id has an invalid or unsafe format")
        if self.concept is not None:
            _bounded_text(self.concept, "concept", _MAX_PROMPT)
        if self.score is not None and not 0 <= _finite(self.score, "score") <= 1:
            raise ValueError("score must be between 0 and 1")
        if self.width != self.mask.width or self.height != self.mask.height:
            raise ValueError("candidate dimensions must match the mask artifact")

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "concept": self.concept,
            "score": self.score,
            "bounding_box": self.bounding_box.as_dict() if self.bounding_box else None,
            "mask": self.mask.as_dict(),
            "width": self.width,
            "height": self.height,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> SegmentationCandidate:
        if not isinstance(value, dict):
            raise ValueError("candidate must be an object")
        box = value.get("bounding_box")
        return cls(
            value.get("candidate_id"),
            value.get("concept"),
            value.get("score"),
            BoundingBox.from_dict(box) if box is not None else None,
            MaskArtifact.from_dict(value.get("mask")),
            value.get("width"),
            value.get("height"),
            value.get("metadata", {}) if isinstance(value.get("metadata", {}), dict) else {},
        )


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    provider: str
    model: str | None
    candidates: tuple[SegmentationCandidate, ...]
    runtime_seconds: float
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    soft_alpha: bool = False

    def __post_init__(self) -> None:
        _bounded_text(self.provider, "provider", 64)
        if len(self.candidates) > _MAX_CANDIDATES:
            raise ValueError("segmentation result contains too many candidates")
        if _finite(self.runtime_seconds, "runtime_seconds") < 0:
            raise ValueError("runtime_seconds cannot be negative")
        if len(self.warnings) > _MAX_WARNINGS or any(
            not isinstance(item, str) for item in self.warnings
        ):
            raise ValueError("warnings are malformed or exceed the safety limit")

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "runtime_seconds": self.runtime_seconds,
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
            "soft_alpha": self.soft_alpha,
        }

    @classmethod
    def from_dict(cls, value: Any) -> SegmentationResult:
        if not isinstance(value, dict):
            raise ValueError("segmentation result must be an object")
        candidates = value.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError("segmentation candidates must be a list")
        warnings = value.get("warnings", [])
        if not isinstance(warnings, list):
            raise ValueError("segmentation warnings must be a list")
        return cls(
            value.get("provider", "unknown"),
            value.get("model"),
            tuple(SegmentationCandidate.from_dict(item) for item in candidates),
            value.get("runtime_seconds"),
            tuple(warnings),
            value.get("provenance", {}) if isinstance(value.get("provenance", {}), dict) else {},
            value.get("soft_alpha") is True,
        )
