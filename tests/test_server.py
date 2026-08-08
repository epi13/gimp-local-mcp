from __future__ import annotations

from gimp_local_mcp import server


def test_server_uses_official_mcp_server_and_registers_tools() -> None:
    assert server.mcp.name == "gimp-local-mcp"
    tools = server.mcp._tool_manager._tools
    assert {
        "gimp_status",
        "list_open_images",
        "get_current_context",
        "get_layer_tree",
        "get_selected_layers",
        "set_selected_layers",
        "create_layer_group",
        "invoke_pdb_procedure",
        "apply_gaussian_blur_filter",
        "apply_brightness_contrast_filter",
        "list_drawable_filters",
    }.issubset(tools)
    assert "execute_scheme" not in tools
    assert "never overwrite files" in server.INSTRUCTIONS
