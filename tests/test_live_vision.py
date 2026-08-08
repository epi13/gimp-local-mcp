from __future__ import annotations

import sys
from pathlib import Path

import pytest

from gimp_local_mcp.config import Config
from gimp_local_mcp.errors import GimpConnectionError
from gimp_local_mcp.gimp.transport import ScriptFuClient
from gimp_local_mcp.service import GimpService

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).parent / "fixtures" / "fake_vision_worker.py"


def test_live_fake_vision_bridge_is_non_destructive() -> None:
    config = Config.from_env()
    probe = ScriptFuClient(config)
    try:
        try:
            probe.connect()
        except GimpConnectionError:
            pytest.skip("GIMP Script-Fu server is not available")
    finally:
        probe.close()
    config = Config(
        host=config.host,
        port=config.port,
        timeout=config.timeout,
        log_level=config.log_level,
        max_response_bytes=config.max_response_bytes,
        allow_remote=config.allow_remote,
        vision_provider="fake",
        vision_command=(sys.executable, str(FIXTURE)),
        vision_timeout=2.0,
    )
    service = GimpService(config)
    image_id: int | None = None
    try:
        image_id = service.create_image(32, 24, "RGB")["image_id"]
        source = service.create_layer(image_id, "Vision source")
        result = service.isolate_subject_vision(
            image_id, source["layer_id"], "synthetic foreground"
        )
        assert result["source_preserved"] is True
        assert result["mask_id"] >= 0
        assert result["quality"]["partial_alpha_ratio"] > 0
        assert service.get_layer_mask_info(image_id, result["working_layer_id"])["has_mask"] is True
    finally:
        if image_id is not None:
            service.close_image(image_id, discard=True)
        service.close()


def test_live_fake_vision_creates_complementary_layer_group() -> None:
    base = Config.from_env()
    config = Config(
        host=base.host,
        port=base.port,
        timeout=base.timeout,
        log_level=base.log_level,
        max_response_bytes=base.max_response_bytes,
        allow_remote=base.allow_remote,
        vision_provider="fake",
        vision_command=(sys.executable, str(FIXTURE)),
        vision_timeout=2.0,
    )
    service = GimpService(config)
    image_id: int | None = None
    try:
        try:
            image_id = service.create_image(32, 24, "RGB")["image_id"]
        except GimpConnectionError:
            pytest.skip("GIMP Script-Fu server is not available")
        source = service.create_layer(image_id, "Original reference")
        source_id = source["layer_id"]
        source_pixel = service._sample_pixel(source_id, 0, 0)
        result = service.separate_subject_to_layers(image_id, source_id, "synthetic subject")

        assert result["source_preserved"] is True
        assert result["generated_layer_count"] == 2
        assert result["group_id"] >= 0
        tree = service.get_layer_tree(image_id)
        group = next(item for item in tree if item["layer_id"] == result["group_id"])
        assert [item["name"] for item in group["children"]] == [
            "Subject — synthetic subject",
            "Background",
        ]
        assert service.get_layer_info(image_id, source_id)["visible"] is False
        assert service._sample_pixel(source_id, 0, 0) == source_pixel
        assert service.get_layer_mask_info(image_id, source_id)["has_mask"] is False
        subject_info, background_info = result["generated_layers"]
        subject_samples = service._sample_mask(subject_info["mask_id"], 32, 24)
        background_samples = service._sample_mask(background_info["mask_id"], 32, 24)
        assert [(x, y) for x, y, _alpha in subject_samples] == [
            (x, y) for x, y, _alpha in background_samples
        ]

        def linear_alpha(value: int) -> float:
            encoded = value / 255
            return encoded / 12.92 if encoded <= 0.04045 else ((encoded + 0.055) / 1.055) ** 2.4

        assert all(
            linear_alpha(subject_alpha) + linear_alpha(background_alpha)
            == pytest.approx(1.0, abs=0.005)
            for (_, _, subject_alpha), (_, _, background_alpha) in zip(
                subject_samples, background_samples, strict=True
            )
        )
        assert result["selected_layer_ids"] == [subject_info["layer_id"]]
    finally:
        if image_id is not None:
            service.close_image(image_id, discard=True)
        service.close()
