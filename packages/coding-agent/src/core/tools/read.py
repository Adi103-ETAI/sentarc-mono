"""Read tool — reads text files and images with truncation support."""

from __future__ import annotations

import asyncio
import base64
import os
from typing import Any, Callable, Dict, List, Optional

from sentarc_coding_agent.core.tools.path_utils import resolve_read_path
from sentarc_coding_agent.core.tools.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    format_size,
    truncate_head,
)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
_MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def _detect_image_mime_type(path: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower()
    return _MIME_MAP.get(ext)


class ReadTool:
    """
    AgentTool for reading files.
    Supports text and images; handles offset/limit for large text files.
    """

    name = "read"
    label = "read"
    description = (
        f"Read the contents of a file. Supports text files and images (jpg, png, gif, webp). "
        f"Images are sent as attachments. For text files, output is truncated to "
        f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
        f"Use offset/limit for large files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
            "offset": {"type": "integer", "description": "Line number to start reading from (1-indexed)"},
            "limit": {"type": "integer", "description": "Maximum number of lines to read"},
        },
        "required": ["path"],
    }

    def __init__(self, cwd: str, auto_resize_images: bool = True):
        self.cwd = cwd
        self.auto_resize_images = auto_resize_images

    async def execute(
        self,
        tool_call_id: str,
        args: Dict[str, Any],
        signal: Optional[asyncio.Event] = None,
        on_update: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        path = args["path"]
        offset: Optional[int] = args.get("offset")
        limit: Optional[int] = args.get("limit")

        absolute_path = resolve_read_path(path, self.cwd)

        if signal and signal.is_set():
            raise Exception("Operation aborted")

        if not os.path.exists(absolute_path):
            raise Exception(f"File not found: {path}")

        if not os.access(absolute_path, os.R_OK):
            raise Exception(f"Permission denied: {path}")

        mime_type = _detect_image_mime_type(absolute_path)

        if mime_type:
            # Read as image
            with open(absolute_path, "rb") as f:
                data = base64.b64encode(f.read()).decode("ascii")
            text_note = f"Read image file [{mime_type}]"
            return {
                "content": [
                    {"type": "text", "text": text_note},
                    {"type": "image", "data": data, "mimeType": mime_type},
                ],
                "details": None,
            }
        else:
            # Read as text
            with open(absolute_path, "r", encoding="utf-8", errors="replace") as f:
                text_content = f.read()

            all_lines = text_content.split("\n")
            total_file_lines = len(all_lines)

            start_line = max(0, (offset - 1)) if offset else 0
            start_line_display = start_line + 1

            if start_line >= len(all_lines):
                raise Exception(
                    f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)"
                )

            user_limited_lines: Optional[int] = None
            if limit is not None:
                end_line = min(start_line + limit, len(all_lines))
                selected_content = "\n".join(all_lines[start_line:end_line])
                user_limited_lines = end_line - start_line
            else:
                selected_content = "\n".join(all_lines[start_line:])

            truncation = truncate_head(selected_content)

            if truncation.first_line_exceeds_limit:
                first_line_size = format_size(len(all_lines[start_line].encode("utf-8")))
                output_text = (
                    f"[Line {start_line_display} is {first_line_size}, exceeds "
                    f"{format_size(DEFAULT_MAX_BYTES)} limit. Use bash: "
                    f"sed -n '{start_line_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
                )
                details = {"truncation": truncation}
            elif truncation.truncated:
                end_line_display = start_line_display + truncation.output_lines - 1
                next_offset = end_line_display + 1
                output_text = truncation.content
                if truncation.truncated_by == "lines":
                    output_text += (
                        f"\n\n[Showing lines {start_line_display}-{end_line_display} of "
                        f"{total_file_lines}. Use offset={next_offset} to continue.]"
                    )
                else:
                    output_text += (
                        f"\n\n[Showing lines {start_line_display}-{end_line_display} of "
                        f"{total_file_lines} ({format_size(DEFAULT_MAX_BYTES)} limit). "
                        f"Use offset={next_offset} to continue.]"
                    )
                details = {"truncation": truncation}
            elif user_limited_lines is not None and start_line + user_limited_lines < len(all_lines):
                remaining = len(all_lines) - (start_line + user_limited_lines)
                next_offset = start_line + user_limited_lines + 1
                output_text = truncation.content
                output_text += f"\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
                details = None
            else:
                output_text = truncation.content
                details = None

            return {
                "content": [{"type": "text", "text": output_text}],
                "details": details,
            }


def create_read_tool(cwd: str, auto_resize_images: bool = True) -> ReadTool:
    return ReadTool(cwd, auto_resize_images)
