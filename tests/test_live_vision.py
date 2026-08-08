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
