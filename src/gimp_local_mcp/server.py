"""MCP server and ergonomic tool registrations."""

from __future__ import annotations

import atexit
import logging
from typing import Any

from mcp.server import MCPServer

from .service import GimpService

logger = logging.getLogger(__name__)

INSTRUCTIONS = """GIMP Local MCP controls a local GIMP 3 instance through its Script-Fu server.
Start uncertain editing workflows with get_current_context, then inspect the recursive layer
tree and selected-layer IDs before targeting a drawable. Prefer ergonomic tools for routine edits,
and use structured PDB discovery/invocation for unsupported operations. Use identifiers
returned by this server. Preserve user work and never overwrite files without explicit
permission. Subject isolation duplicates its explicit source layer and applies a bounded layer
mask. Configured local vision workers support semantic prompts such as “red fox”; otherwise auto
mode uses the bounded high-key fallback. Neither mode claims semantic segmentation accuracy.
GIMP and vision inference remain local; this server does not generate replacement pixels or send
images to cloud services."""

mcp = MCPServer(
    "gimp-local-mcp",
    version="0.1.0",
    instructions=INSTRUCTIONS,
    log_level="INFO",
)
_service = GimpService()
atexit.register(_service.close)


@mcp.tool()
def gimp_status() -> dict[str, Any]:
    """Check the configured GIMP connection and return basic session status."""

    return _service.status()


@mcp.tool()
def gimp_capabilities() -> dict[str, Any]:
    """Return supported high-level operations and known v0.1 limitations."""

    return _service.capabilities()


@mcp.tool()
def list_open_images() -> list[dict[str, Any]]:
    """List open images using stable GIMP image IDs."""

    return _service.list_open_images()


@mcp.tool()
def get_active_image() -> dict[str, Any] | None:
    """Return the current image, or the sole open image when no display context is exposed."""

    return _service.get_active_image()


@mcp.tool()
def get_current_context() -> dict[str, Any]:
    """Return a concise current-image and multi-layer selection snapshot for editing."""

    return _service.get_current_context()


@mcp.tool()
def get_image_info(image_id: int) -> dict[str, Any]:
    """Return dimensions, name, file, base type, and dirty state for an image."""

    return _service.get_image_info(image_id)


@mcp.tool()
def open_image(path: str) -> dict[str, Any]:
    """Open an existing local image file in GIMP."""

    return _service.open_image(path)


@mcp.tool()
def create_image(width: int, height: int, base_type: str = "RGB") -> dict[str, Any]:
    """Create a new RGB, GRAY, or INDEXED image."""

    return _service.create_image(width, height, base_type)


@mcp.tool()
def save_xcf(image_id: int, path: str, overwrite: bool = False) -> dict[str, Any]:
    """Save an image as XCF; existing files require overwrite=true."""

    return _service.save_xcf(image_id, path, overwrite)


@mcp.tool()
def export_image(image_id: int, path: str, overwrite: bool = False) -> dict[str, Any]:
    """Export an image using the format selected by the explicit file extension."""

    return _service.export_image(image_id, path, overwrite)


@mcp.tool()
def close_image(image_id: int, discard: bool = False) -> dict[str, Any]:
    """Close an image; dirty images require an explicit discard=true."""

    return _service.close_image(image_id, discard)


@mcp.tool()
def list_layers(image_id: int) -> list[dict[str, Any]]:
    """List root layers in top-to-bottom order."""

    return _service.list_layers(image_id)


@mcp.tool()
def get_layer_tree(
    image_id: int, max_depth: int = 32, max_items: int = 1000
) -> list[dict[str, Any]]:
    """Return the bounded recursive layer and group hierarchy for an image."""

    return _service.get_layer_tree(image_id, max_depth=max_depth, max_items=max_items)


@mcp.tool()
def get_selected_layers(image_id: int) -> list[dict[str, Any]]:
    """Return all selected layers; GIMP 3 supports multi-layer selection."""

    return _service.get_selected_layers(image_id)


@mcp.tool()
def set_selected_layers(image_id: int, layer_ids: list[int]) -> dict[str, Any]:
    """Set selected layers and return the state read back from GIMP."""

    return _service.set_selected_layers(image_id, layer_ids)


@mcp.tool()
def get_layer_info(image_id: int, layer_id: int) -> dict[str, Any]:
    """Return stable properties for a layer/drawable."""

    return _service.get_layer_info(image_id, layer_id)


@mcp.tool()
def create_layer(
    image_id: int,
    name: str,
    width: int | None = None,
    height: int | None = None,
    opacity: float = 100.0,
    parent_id: int | None = None,
) -> dict[str, Any]:
    """Create a transparent RGBA layer at the top of an image or group."""

    return _service.create_layer(image_id, name, width, height, opacity, parent_id)


@mcp.tool()
def create_layer_group(image_id: int, name: str, parent_id: int | None = None) -> dict[str, Any]:
    """Create a layer group at the top of an image or existing group."""

    return _service.create_layer_group(image_id, name, parent_id)


@mcp.tool()
def duplicate_layer(image_id: int, layer_id: int) -> dict[str, Any]:
    """Duplicate a layer and insert the copy at the top of the image."""

    return _service.duplicate_layer(image_id, layer_id)


@mcp.tool()
def rename_layer(layer_id: int, name: str) -> dict[str, Any]:
    """Rename a layer without changing its pixels."""

    return _service.rename_layer(layer_id, name)


@mcp.tool()
def delete_layer(image_id: int, layer_id: int) -> dict[str, Any]:
    """Delete a layer from an image."""

    return _service.delete_layer(image_id, layer_id)


@mcp.tool()
def set_layer_visibility(layer_id: int, visible: bool) -> dict[str, Any]:
    """Set whether a layer is visible."""

    return _service.set_layer_visibility(layer_id, visible)


@mcp.tool()
def set_layer_opacity(layer_id: int, opacity: float) -> dict[str, Any]:
    """Set layer opacity from 0 to 100."""

    return _service.set_layer_opacity(layer_id, opacity)


@mcp.tool()
def set_layer_mode(layer_id: int, mode: str) -> dict[str, Any]:
    """Set a GIMP layer mode such as LAYER-MODE-NORMAL or LAYER-MODE-MULTIPLY."""

    return _service.set_layer_mode(layer_id, mode)


@mcp.tool()
def move_layer(
    image_id: int, layer_id: int, position: int, parent_id: int | None = None
) -> dict[str, Any]:
    """Move a layer to a top-level or group position."""

    return _service.move_layer(image_id, layer_id, position, parent_id)


@mcp.tool()
def merge_down(
    image_id: int, layer_id: int, merge_type: str = "EXPAND-AS-NECESSARY"
) -> dict[str, Any]:
    """Merge a layer down using a GIMP merge type."""

    return _service.merge_down(image_id, layer_id, merge_type)


@mcp.tool()
def resize_image(image_id: int, width: int, height: int) -> dict[str, Any]:
    """Scale the image and its layers to exact dimensions."""

    return _service.resize_image(image_id, width, height)


@mcp.tool()
def resize_canvas(
    image_id: int, width: int, height: int, offset_x: int = 0, offset_y: int = 0
) -> dict[str, Any]:
    """Resize the canvas without scaling layer pixels."""

    return _service.resize_canvas(image_id, width, height, offset_x, offset_y)


@mcp.tool()
def crop_image(
    image_id: int, width: int, height: int, offset_x: int = 0, offset_y: int = 0
) -> dict[str, Any]:
    """Crop an image to the supplied rectangle."""

    return _service.crop_image(image_id, width, height, offset_x, offset_y)


@mcp.tool()
def rotate_image(image_id: int, rotation: str) -> dict[str, Any]:
    """Rotate an image using ROTATE-90, ROTATE-180, or ROTATE-270."""

    return _service.rotate(image_id, rotation)


@mcp.tool()
def flip_image(image_id: int, direction: str) -> dict[str, Any]:
    """Flip an image HORIZONTAL or VERTICAL."""

    return _service.flip(image_id, direction)


@mcp.tool()
def select_all(image_id: int) -> dict[str, Any]:
    """Select the entire image."""

    return _service.select_all(image_id)


@mcp.tool()
def select_none(image_id: int) -> dict[str, Any]:
    """Clear the image selection."""

    return _service.select_none(image_id)


@mcp.tool()
def invert_selection(image_id: int) -> dict[str, Any]:
    """Invert the current selection."""

    return _service.invert_selection(image_id)


@mcp.tool()
def select_rectangle(image_id: int, x: int, y: int, width: int, height: int) -> dict[str, Any]:
    """Replace the selection with a rectangle."""

    return _service.select_rectangle(image_id, x, y, width, height)


@mcp.tool()
def select_ellipse(image_id: int, x: int, y: int, width: int, height: int) -> dict[str, Any]:
    """Replace the selection with an ellipse."""

    return _service.select_ellipse(image_id, x, y, width, height)


@mcp.tool()
def select_layer_alpha(image_id: int, layer_id: int) -> dict[str, Any]:
    """Replace the selection with a layer's alpha."""

    return _service.select_layer_alpha(image_id, layer_id)


@mcp.tool()
def get_layer_mask_info(image_id: int, layer_id: int) -> dict[str, Any]:
    """Return the mask identity and enabled/visible/editable state for a layer."""

    return _service.get_layer_mask_info(image_id, layer_id)


@mcp.tool()
def create_layer_mask(image_id: int, layer_id: int, mask_type: str = "selection") -> dict[str, Any]:
    """Create and attach one bounded GIMP layer mask; existing masks are never replaced."""

    return _service.create_layer_mask(image_id, layer_id, mask_type)


@mcp.tool()
def set_layer_mask_enabled(image_id: int, layer_id: int, enabled: bool) -> dict[str, Any]:
    """Enable or disable application of an existing layer mask."""

    return _service.set_layer_mask_enabled(image_id, layer_id, enabled)


@mcp.tool()
def vision_status() -> dict[str, Any]:
    """Report the configured local vision provider and its capabilities."""

    return _service.vision_status()


@mcp.tool()
def segment_subject(
    image_id: int,
    layer_id: int,
    prompt: str,
    max_candidates: int = 3,
    minimum_score: float = 0.0,
    mode: str = "semantic",
) -> dict[str, Any]:
    """Run local semantic segmentation against a temporary current-GIMP snapshot."""

    return _service.segment_subject(
        image_id,
        layer_id,
        prompt,
        max_candidates=max_candidates,
        minimum_score=minimum_score,
        mode=mode,
    )


@mcp.tool()
def isolate_subject_vision(
    image_id: int,
    layer_id: int,
    prompt: str,
    max_candidates: int = 3,
    minimum_score: float = 0.0,
    mode: str = "semantic",
) -> dict[str, Any]:
    """Duplicate a layer and apply a local semantic mask non-destructively."""

    return _service.isolate_subject_vision(
        image_id,
        layer_id,
        prompt,
        max_candidates=max_candidates,
        minimum_score=minimum_score,
        mode=mode,
    )


@mcp.tool()
def isolate_subject(
    image_id: int,
    layer_id: int,
    strategy: str = "auto",
    background_threshold: int = 48,
    refinement_threshold: int = 24,
    feather: int = 1,
    prompt: str = "subject",
) -> dict[str, Any]:
    """Use local semantic vision when available, otherwise the bounded high-key fallback."""

    return _service.isolate_subject(
        image_id,
        layer_id,
        strategy,
        background_threshold,
        refinement_threshold,
        feather,
        prompt,
    )


@mcp.tool()
def brightness_contrast(layer_id: int, brightness: float, contrast: float) -> dict[str, Any]:
    """Apply the basic brightness/contrast adjustment to a drawable."""

    return _service.brightness_contrast(layer_id, brightness, contrast)


@mcp.tool()
def apply_gaussian_blur_filter(
    layer_id: int,
    radius_x: float,
    radius_y: float | None = None,
    name: str = "Gaussian Blur",
    opacity: float = 100.0,
    blend_mode: str = "LAYER-MODE-REPLACE",
) -> dict[str, Any]:
    """Add a non-destructive Gaussian blur GEGL filter to a layer."""

    return _service.apply_gaussian_blur_filter(
        layer_id,
        radius_x,
        radius_y,
        name=name,
        opacity=opacity,
        blend_mode=blend_mode,
    )


@mcp.tool()
def apply_brightness_contrast_filter(
    layer_id: int,
    brightness: float,
    contrast: float,
    name: str = "Brightness / Contrast",
    opacity: float = 100.0,
    blend_mode: str = "LAYER-MODE-REPLACE",
) -> dict[str, Any]:
    """Add a non-destructive GEGL brightness/contrast filter to a layer."""

    return _service.apply_brightness_contrast_filter(
        layer_id,
        brightness,
        contrast,
        name=name,
        opacity=opacity,
        blend_mode=blend_mode,
    )


@mcp.tool()
def list_drawable_filters(layer_id: int) -> list[dict[str, Any]]:
    """List non-destructive filters GIMP reports on a layer."""

    return _service.list_drawable_filters(layer_id)


@mcp.tool()
def hue_saturation(
    layer_id: int, hue: float, saturation: float, lightness: float
) -> dict[str, Any]:
    """Apply the basic hue, saturation, and lightness adjustment."""

    return _service.hue_saturation(layer_id, hue, saturation, lightness)


@mcp.tool()
def desaturate(layer_id: int, mode: str = "DESATURATE-LUMA") -> dict[str, Any]:
    """Desaturate a drawable using a GIMP desaturation mode."""

    return _service.desaturate(layer_id, mode)


@mcp.tool()
def undo(image_id: int) -> dict[str, Any]:
    """Undo the latest action in an image."""

    return _service.undo(image_id)


@mcp.tool()
def redo(image_id: int) -> dict[str, Any]:
    """Redo the latest undone action in an image."""

    return _service.redo(image_id)


@mcp.tool()
def search_pdb(query: str, limit: int = 50) -> list[str]:
    """Search live GIMP PDB procedure names by literal text."""

    return _service.catalog.search(query, limit=limit)


@mcp.tool()
def describe_pdb_procedure(procedure_name: str) -> dict[str, Any]:
    """Describe a live GIMP PDB procedure using runtime metadata."""

    return _service.catalog.describe(procedure_name).as_dict()


@mcp.tool()
def invoke_pdb_procedure(
    procedure_name: str,
    args: list[Any] | None = None,
    keywords: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoke a live PDB procedure using structured positional or keyword values.

    Values may be JSON scalars, arrays, or {"scheme_symbol": "ENUM-VALUE"} objects.
    Arbitrary Scheme source is intentionally not accepted.
    """

    return _service.pdb.invoke(procedure_name, args or [], keywords=keywords)


def run() -> None:
    """Run the server over MCP stdio."""

    mcp.run()


if __name__ == "__main__":
    run()
