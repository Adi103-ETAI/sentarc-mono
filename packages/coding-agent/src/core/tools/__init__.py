"""Tool registry — creates and exports all agent tools."""

from __future__ import annotations

from typing import Dict, List, Optional

from sentarc_coding_agent.core.tools.bash import BashTool, create_bash_tool
from sentarc_coding_agent.core.tools.edit import EditTool, create_edit_tool
from sentarc_coding_agent.core.tools.find import FindTool, create_find_tool
from sentarc_coding_agent.core.tools.grep import GrepTool, create_grep_tool
from sentarc_coding_agent.core.tools.ls import LsTool, create_ls_tool
from sentarc_coding_agent.core.tools.read import ReadTool, create_read_tool
from sentarc_coding_agent.core.tools.write import WriteTool, create_write_tool

# All available tool names
TOOL_NAMES = ("read", "bash", "edit", "write", "grep", "find", "ls")


def create_coding_tools(cwd: str) -> List:
    """Create default coding tools (read, bash, edit, write)."""
    return [
        create_read_tool(cwd),
        create_bash_tool(cwd),
        create_edit_tool(cwd),
        create_write_tool(cwd),
    ]


def create_read_only_tools(cwd: str) -> List:
    """Create read-only tools (read, grep, find, ls)."""
    return [
        create_read_tool(cwd),
        create_grep_tool(cwd),
        create_find_tool(cwd),
        create_ls_tool(cwd),
    ]


def create_all_tools(cwd: str) -> Dict[str, object]:
    """Create all tools configured for a specific working directory."""
    return {
        "read": create_read_tool(cwd),
        "bash": create_bash_tool(cwd),
        "edit": create_edit_tool(cwd),
        "write": create_write_tool(cwd),
        "grep": create_grep_tool(cwd),
        "find": create_find_tool(cwd),
        "ls": create_ls_tool(cwd),
    }


def create_tools(cwd: str, tool_names: Optional[List[str]] = None) -> List:
    """Create a subset of tools by name."""
    if tool_names is None:
        tool_names = ["read", "bash", "edit", "write"]

    all_tools = create_all_tools(cwd)
    result = []
    for name in tool_names:
        tool = all_tools.get(name)
        if tool:
            result.append(tool)
    return result


__all__ = [
    "BashTool",
    "EditTool",
    "FindTool",
    "GrepTool",
    "LsTool",
    "ReadTool",
    "WriteTool",
    "TOOL_NAMES",
    "create_coding_tools",
    "create_read_only_tools",
    "create_all_tools",
    "create_tools",
    "create_bash_tool",
    "create_edit_tool",
    "create_find_tool",
    "create_grep_tool",
    "create_ls_tool",
    "create_read_tool",
    "create_write_tool",
]
