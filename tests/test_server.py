from __future__ import annotations

from gimp_local_mcp import server


def test_server_uses_official_mcp_server_and_registers_tools() -> None:
    assert server.mcp.name == "gimp-local-mcp"
    tools = server.mcp._tool_manager._tools
    assert {"gimp_status", "list_open_images", "invoke_pdb_procedure"}.issubset(tools)
    assert "execute_scheme" not in tools
    assert "never overwrite files" in server.INSTRUCTIONS
