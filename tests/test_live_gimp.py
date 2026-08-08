from __future__ import annotations

import pytest

from gimp_local_mcp.config import Config
from gimp_local_mcp.errors import GimpConnectionError
from gimp_local_mcp.gimp.scheme import parse_scheme, unwrap
from gimp_local_mcp.gimp.transport import ScriptFuClient
from gimp_local_mcp.service import GimpService


@pytest.mark.integration
def test_live_gimp_version() -> None:
    config = Config.from_env()
    probe = ScriptFuClient(config)
    try:
        try:
            probe.connect()
        except GimpConnectionError:
            pytest.skip("GIMP Script-Fu server is not available")
        version = probe.execute("(gimp-version)")
    finally:
        probe.close()
    assert unwrap(parse_scheme(version)) == "3.2.0"


@pytest.mark.integration
def test_live_gimp_non_destructive_filters() -> None:
    config = Config.from_env()
    service = GimpService(config)
    image_id: int | None = None
    try:
        assert service.status()["gimp_version"] == "3.2.0"
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
        assert service.get_image_info(image_id)["width"] == 16
        assert service.get_image_info(image_id)["height"] == 16

        with pytest.raises(ValueError):
            service.apply_gaussian_blur_filter(layer["layer_id"], float("nan"))
    finally:
        if image_id is not None:
            service.close_image(image_id, discard=True)
        service.close()
