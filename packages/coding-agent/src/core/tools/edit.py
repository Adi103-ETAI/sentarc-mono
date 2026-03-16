"""Edit tool — surgical find-and-replace edits in files."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable, Dict, Optional

from sentarc_coding_agent.core.tools.edit_diff import (
    detect_line_ending,
    fuzzy_find_text,
    generate_diff_string,
    normalize_for_fuzzy_match,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from sentarc_coding_agent.core.tools.path_utils import resolve_to_cwd


class EditTool:
    """AgentTool for surgical file edits."""

    name = "edit"
    label = "edit"
    description = (
        "Edit a file by replacing exact text. The oldText must match exactly "
        "(including whitespace). Use this for precise, surgical edits."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit (relative or absolute)"},
            "oldText": {"type": "string", "description": "Exact text to find and replace (must match exactly)"},
            "newText": {"type": "string", "description": "New text to replace the old text with"},
        },
        "required": ["path", "oldText", "newText"],
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
        old_text = args["oldText"]
        new_text = args["newText"]

        if signal and signal.is_set():
            raise Exception("Operation aborted")

        absolute_path = resolve_to_cwd(path, self.cwd)

        if not os.path.exists(absolute_path):
            raise Exception(f"File not found: {path}")

        if not os.access(absolute_path, os.R_OK | os.W_OK):
            raise Exception(f"File not readable/writable: {path}")

        with open(absolute_path, "r", encoding="utf-8", errors="replace") as f:
            raw_content = f.read()

        if signal and signal.is_set():
            raise Exception("Operation aborted")

        bom_result = strip_bom(raw_content)
        bom = bom_result.bom
        content = bom_result.text

        original_ending = detect_line_ending(content)
        normalized_content = normalize_to_lf(content)
        normalized_old_text = normalize_to_lf(old_text)
        normalized_new_text = normalize_to_lf(new_text)

        match_result = fuzzy_find_text(normalized_content, normalized_old_text)

        if not match_result.found:
            raise Exception(
                f"Could not find the exact text in {path}. "
                f"The old text must match exactly including all whitespace and newlines."
            )

        # Count occurrences using fuzzy-normalized content
        fuzzy_content = normalize_for_fuzzy_match(normalized_content)
        fuzzy_old = normalize_for_fuzzy_match(normalized_old_text)
        occurrences = fuzzy_content.count(fuzzy_old)

        if occurrences > 1:
            raise Exception(
                f"Found {occurrences} occurrences of the text in {path}. "
                f"The text must be unique. Please provide more context to make it unique."
            )

        if signal and signal.is_set():
            raise Exception("Operation aborted")

        base_content = match_result.content_for_replacement
        new_content = (
            base_content[:match_result.index]
            + normalized_new_text
            + base_content[match_result.index + match_result.match_length:]
        )

        if base_content == new_content:
            raise Exception(
                f"No changes made to {path}. The replacement produced identical content."
            )

        final_content = bom + restore_line_endings(new_content, original_ending)

        with open(absolute_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        diff_result = generate_diff_string(base_content, new_content)

        return {
            "content": [{"type": "text", "text": f"Successfully replaced text in {path}."}],
            "details": {
                "diff": diff_result.diff,
                "firstChangedLine": diff_result.first_changed_line,
            },
        }


def create_edit_tool(cwd: str) -> EditTool:
    return EditTool(cwd)
