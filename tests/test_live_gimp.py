from __future__ import annotations

import socket

import pytest

from gimp_local_mcp.config import Config
from gimp_local_mcp.gimp.transport import ScriptFuClient


@pytest.mark.integration
def test_live_gimp_version() -> None:
    config = Config.from_env()
    probe = socket.socket()
    probe.settimeout(0.2)
    try:
        try:
            probe.connect((config.host, config.port))
        except OSError:
            pytest.skip("GIMP Script-Fu server is not available")
    finally:
        probe.close()
    with ScriptFuClient(config) as client:
        assert client.execute("(gimp-version)")
