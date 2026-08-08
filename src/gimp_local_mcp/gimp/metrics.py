"""Observable, ground-truth-free mask quality proxies."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def mask_metrics(
    samples: Iterable[tuple[int, int, int]],
    width: int,
    height: int,
    *,
    foreground_reference: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Summarize bounded mask samples; values are proxies, not accuracy."""

    if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
        raise ValueError("mask dimensions must be positive integers")
    points = list(samples)
    if not points:
        return {
            "sample_count": 0,
            "empty_sample": True,
            "all_transparent": True,
            "all_opaque": False,
            "border_transparency_ratio": None,
            "transparent_ratio": 0.0,
            "opaque_ratio": 0.0,
            "partial_alpha_ratio": 0.0,
            "edge_transition_sample_ratio": 0.0,
            "mask_coverage": 0.0,
            "retained_bounding_box": None,
            "foreground_retention_proxy": None,
        }
    for x, y, alpha in points:
        if not 0 <= x < width or not 0 <= y < height or not 0 <= alpha <= 255:
            raise ValueError("mask sample is outside dimensions or alpha range")
    transparent = [alpha <= 8 for _, _, alpha in points]
    opaque = [alpha >= 247 for _, _, alpha in points]
    partial = [not t and not o for t, o in zip(transparent, opaque, strict=True)]
    border = [
        t
        for (x, y, _), t in zip(points, transparent, strict=True)
        if x in {0, width - 1} or y in {0, height - 1}
    ]
    retained = [(x, y) for (x, y, alpha) in points if alpha > 8]
    point_map = {(x, y): alpha for x, y, alpha in points}
    x_values = sorted({x for x, _, _ in points})
    y_values = sorted({y for _, y, _ in points})
    edge_partial = 0
    for x, y, alpha in points:
        if not 8 < alpha < 247:
            continue
        x_index, y_index = x_values.index(x), y_values.index(y)
        neighbors = []
        if x_index:
            neighbors.append(point_map.get((x_values[x_index - 1], y)))
        if x_index + 1 < len(x_values):
            neighbors.append(point_map.get((x_values[x_index + 1], y)))
        if y_index:
            neighbors.append(point_map.get((x, y_values[y_index - 1])))
        if y_index + 1 < len(y_values):
            neighbors.append(point_map.get((x, y_values[y_index + 1])))
        if any(value is not None and (value <= 8 or value >= 247) for value in neighbors):
            edge_partial += 1
    foreground_retention = None
    if foreground_reference is not None:
        if len(foreground_reference) != len(points):
            raise ValueError("foreground reference must align with mask samples")
        confident = [value >= 247 for value in foreground_reference]
        retained_confident = [
            alpha > 8
            for alpha, is_confident in zip((p[2] for p in points), confident, strict=True)
            if is_confident
        ]
        foreground_retention = (
            sum(retained_confident) / len(retained_confident) if retained_confident else None
        )
    return {
        "sample_count": len(points),
        "empty_sample": False,
        "all_transparent": all(transparent),
        "all_opaque": all(opaque),
        "border_transparency_ratio": sum(border) / len(border) if border else None,
        "transparent_ratio": sum(transparent) / len(points),
        "opaque_ratio": sum(opaque) / len(points),
        "partial_alpha_ratio": sum(partial) / len(points),
        "edge_transition_sample_ratio": edge_partial / len(points),
        "mask_coverage": len(retained) / len(points),
        "retained_bounding_box": (
            {
                "x": min(x for x, _ in retained),
                "y": min(y for _, y in retained),
                "width": max(x for x, _ in retained) - min(x for x, _ in retained) + 1,
                "height": max(y for _, y in retained) - min(y for _, y in retained) + 1,
                "sampled": True,
            }
            if retained
            else None
        ),
        "foreground_retention_proxy": foreground_retention,
    }
