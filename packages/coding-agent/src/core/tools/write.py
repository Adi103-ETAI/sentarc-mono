"""Write tool — creates or overwrites files, creating parent directories as needed."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, Dict, Optional

from sentarc_coding_agent.core.tools.path_utils import resolve_to_cwd


class WriteTool:
    """AgentTool for writing files."""

    name = "write"
    label = "write"
    description = (
        "Write content to a file. Creates the file if it doesn't exist, "
        "overwrites if it does. Automatically creates parent directories."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
            "content": {"type": "string", "description": "Content to write to the file"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, cwd: str):
        self.cwd = cwd

    async def execute(
        self,
        tool_call_id: str,
        args: Dict[str, Any],
        signal: Optional[asyncio.Event] = None,
        on_update: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        path = args["path"]
        content = args["content"]

        if signal and signal.is_set():
            raise Exception("Operation aborted")

        absolute_path = resolve_to_cwd(path, self.cwd)
        parent_dir = os.path.dirname(absolute_path)

        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        if signal and signal.is_set():
            raise Exception("Operation aborted")

        with open(absolute_path, "w", encoding="utf-8") as f:
            f.write(content)

        return {
            "content": [{"type": "text", "text": f"Successfully wrote {len(content)} bytes to {path}"}],
            "details": None,
        }


def create_write_tool(cwd: str) -> WriteTool:
    return WriteTool(cwd)
