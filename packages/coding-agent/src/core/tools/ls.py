"""Ls tool — lists directory contents."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, Dict, List, Optional

from sentarc_coding_agent.core.tools.path_utils import resolve_to_cwd
from sentarc_coding_agent.core.tools.truncate import (
    DEFAULT_MAX_BYTES,
    TruncationOptions,
    format_size,
    truncate_head,
)

DEFAULT_LIMIT = 500


class LsTool:
    """AgentTool for listing directory contents."""

    name = "ls"
    label = "ls"
    description = (
        f"List directory contents. Returns entries sorted alphabetically, with '/' suffix "
        f"for directories. Includes dotfiles. Output is truncated to {DEFAULT_LIMIT} entries "
        f"or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list (default: current directory)"},
            "limit": {"type": "integer", "description": f"Maximum number of entries to return (default: {DEFAULT_LIMIT})"},
        },
        "required": [],
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
        path: Optional[str] = args.get("path")
        limit: int = args.get("limit") or DEFAULT_LIMIT

        if signal and signal.is_set():
            raise Exception("Operation aborted")

        dir_path = resolve_to_cwd(path or ".", self.cwd)

        if not os.path.exists(dir_path):
            raise Exception(f"Path not found: {dir_path}")

        if not os.path.isdir(dir_path):
            raise Exception(f"Not a directory: {dir_path}")

        try:
            entries = os.listdir(dir_path)
        except PermissionError as e:
            raise Exception(f"Cannot read directory: {e}")

        # Sort alphabetically, case-insensitive
        entries.sort(key=lambda x: x.lower())

        results: List[str] = []
        entry_limit_reached = False

        for entry in entries:
            if len(results) >= limit:
                entry_limit_reached = True
                break

            full_path = os.path.join(dir_path, entry)
            try:
                suffix = "/" if os.path.isdir(full_path) else ""
            except Exception:
                continue

            results.append(entry + suffix)

        if signal and signal.is_set():
            raise Exception("Operation aborted")

        if not results:
            return {"content": [{"type": "text", "text": "(empty directory)"}], "details": None}

        raw_output = "\n".join(results)
        truncation = truncate_head(raw_output, TruncationOptions(max_lines=2**31 - 1))

        output = truncation.content
        details: Dict[str, Any] = {}
        notices: List[str] = []

        if entry_limit_reached:
            notices.append(f"{limit} entries limit reached. Use limit={limit * 2} for more")
            details["entryLimitReached"] = limit

        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details["truncation"] = truncation

        if notices:
            output += f"\n\n[{'. '.join(notices)}]"

        return {
            "content": [{"type": "text", "text": output}],
            "details": details if details else None,
        }


def create_ls_tool(cwd: str) -> LsTool:
    return LsTool(cwd)
