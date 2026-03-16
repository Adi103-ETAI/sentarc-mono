"""Grep tool — searches file contents using ripgrep or Python fallback."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from sentarc_coding_agent.core.tools.path_utils import resolve_to_cwd
from sentarc_coding_agent.core.tools.truncate import (
    DEFAULT_MAX_BYTES,
    GREP_MAX_LINE_LENGTH,
    TruncationOptions,
    format_size,
    truncate_head,
    truncate_line,
)

DEFAULT_LIMIT = 100


class GrepTool:
    """AgentTool for searching file contents."""

    name = "grep"
    label = "grep"
    description = (
        f"Search file contents for a pattern. Returns matching lines with file paths "
        f"and line numbers. Respects .gitignore. Output is truncated to {DEFAULT_LIMIT} "
        f"matches or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
        f"Long lines are truncated to {GREP_MAX_LINE_LENGTH} chars."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Search pattern (regex or literal string)"},
            "path": {"type": "string", "description": "Directory or file to search (default: current directory)"},
            "glob": {"type": "string", "description": "Filter files by glob pattern, e.g. '*.ts'"},
            "ignoreCase": {"type": "boolean", "description": "Case-insensitive search (default: false)"},
            "literal": {"type": "boolean", "description": "Treat pattern as literal string (default: false)"},
            "context": {"type": "integer", "description": "Lines of context before and after each match (default: 0)"},
            "limit": {"type": "integer", "description": f"Maximum number of matches (default: {DEFAULT_LIMIT})"},
        },
        "required": ["pattern"],
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
        pattern: str = args["pattern"]
        search_dir: Optional[str] = args.get("path")
        glob_pattern: Optional[str] = args.get("glob")
        ignore_case: bool = args.get("ignoreCase", False)
        literal: bool = args.get("literal", False)
        context: int = max(0, args.get("context") or 0)
        limit: int = max(1, args.get("limit") or DEFAULT_LIMIT)

        if signal and signal.is_set():
            raise Exception("Operation aborted")

        search_path = resolve_to_cwd(search_dir or ".", self.cwd)

        if not os.path.exists(search_path):
            raise Exception(f"Path not found: {search_path}")

        is_directory = os.path.isdir(search_path)

        rg_path = _find_rg()
        if rg_path:
            result = await _run_rg(
                rg_path, pattern, search_path, glob_pattern, ignore_case, literal, context, limit, signal
            )
        else:
            result = _python_grep(
                pattern, search_path, glob_pattern, ignore_case, literal, context, limit
            )

        matches = result["matches"]
        match_limit_reached = result["match_limit_reached"]
        lines_truncated = result["lines_truncated"]
        is_directory_res = result.get("is_directory", is_directory)

        if not matches:
            return {"content": [{"type": "text", "text": "No matches found"}], "details": None}

        # Format output lines
        output_lines: List[str] = []
        for m in matches:
            file_path = m["file"]
            line_num = m["line"]
            line_text = m["text"]
            is_match = m.get("is_match", True)

            if is_directory_res:
                rel = os.path.relpath(file_path, search_path).replace(os.sep, "/")
                display_path = rel if not rel.startswith("..") else os.path.basename(file_path)
            else:
                display_path = os.path.basename(file_path)

            trunc_text, was_truncated = truncate_line(line_text)
            if was_truncated:
                lines_truncated = True

            sep = ":" if is_match else "-"
            output_lines.append(f"{display_path}{sep}{line_num}{sep} {trunc_text}")

        raw_output = "\n".join(output_lines)
        truncation = truncate_head(raw_output, TruncationOptions(max_lines=2**31 - 1))

        output = truncation.content
        details: Dict[str, Any] = {}
        notices: List[str] = []

        if match_limit_reached:
            notices.append(
                f"{limit} matches limit reached. Use limit={limit * 2} for more, or refine pattern"
            )
            details["matchLimitReached"] = limit

        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details["truncation"] = truncation

        if lines_truncated:
            notices.append(
                f"Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. Use read tool to see full lines"
            )
            details["linesTruncated"] = True

        if notices:
            output += f"\n\n[{'. '.join(notices)}]"

        return {
            "content": [{"type": "text", "text": output}],
            "details": details if details else None,
        }


def _find_rg() -> Optional[str]:
    import shutil
    return shutil.which("rg")


async def _run_rg(
    rg_path: str,
    pattern: str,
    search_path: str,
    glob_pattern: Optional[str],
    ignore_case: bool,
    literal: bool,
    context: int,
    limit: int,
    signal: Optional[asyncio.Event],
) -> Dict[str, Any]:
    """Run ripgrep and parse JSON output."""
    import json

    args = ["--json", "--line-number", "--color=never", "--hidden"]
    if ignore_case:
        args.append("--ignore-case")
    if literal:
        args.append("--fixed-strings")
    if glob_pattern:
        args += ["--glob", glob_pattern]
    if context > 0:
        args += ["-C", str(context)]
    args += [pattern, search_path]

    try:
        proc = await asyncio.create_subprocess_exec(
            rg_path, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
    except asyncio.TimeoutError:
        return {"matches": [], "match_limit_reached": False, "lines_truncated": False}
    except Exception:
        return {"matches": [], "match_limit_reached": False, "lines_truncated": False}

    output = stdout.decode("utf-8", errors="replace")
    matches: List[Dict[str, Any]] = []
    match_count = 0
    match_limit_reached = False

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if event.get("type") == "match":
            data = event.get("data", {})
            file_path = data.get("path", {}).get("text", "")
            line_number = data.get("line_number", 0)
            line_text = data.get("lines", {}).get("text", "").rstrip("\n")

            matches.append({
                "file": file_path,
                "line": line_number,
                "text": line_text,
                "is_match": True,
            })
            match_count += 1

            # Also add context lines if present
            for subm in data.get("submatches", []):
                pass  # submatches are part of the match line

            if match_count >= limit:
                match_limit_reached = True
                break
        elif event.get("type") == "context":
            data = event.get("data", {})
            file_path = data.get("path", {}).get("text", "")
            line_number = data.get("line_number", 0)
            line_text = data.get("lines", {}).get("text", "").rstrip("\n")
            matches.append({
                "file": file_path,
                "line": line_number,
                "text": line_text,
                "is_match": False,
            })

    return {
        "matches": matches,
        "match_limit_reached": match_limit_reached,
        "lines_truncated": False,
        "is_directory": os.path.isdir(search_path),
    }


def _python_grep(
    pattern: str,
    search_path: str,
    glob_pattern: Optional[str],
    ignore_case: bool,
    literal: bool,
    context: int,
    limit: int,
) -> Dict[str, Any]:
    """Python-based grep fallback."""
    import fnmatch

    flags = re.IGNORECASE if ignore_case else 0
    if literal:
        compiled = re.compile(re.escape(pattern), flags)
    else:
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            raise Exception(f"Invalid regex pattern: {e}")

    matches: List[Dict[str, Any]] = []
    match_count = 0
    match_limit_reached = False

    def search_file(file_path: str) -> bool:
        nonlocal match_count, match_limit_reached
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception:
            return True

        for i, line in enumerate(lines):
            line_text = line.rstrip("\n\r")
            if compiled.search(line_text):
                # Add context before
                if context > 0:
                    for j in range(max(0, i - context), i):
                        matches.append({
                            "file": file_path,
                            "line": j + 1,
                            "text": lines[j].rstrip("\n\r"),
                            "is_match": False,
                        })
                matches.append({
                    "file": file_path,
                    "line": i + 1,
                    "text": line_text,
                    "is_match": True,
                })
                # Add context after
                if context > 0:
                    for j in range(i + 1, min(len(lines), i + 1 + context)):
                        matches.append({
                            "file": file_path,
                            "line": j + 1,
                            "text": lines[j].rstrip("\n\r"),
                            "is_match": False,
                        })
                match_count += 1
                if match_count >= limit:
                    match_limit_reached = True
                    return False
        return True

    if os.path.isfile(search_path):
        search_file(search_path)
    else:
        for dirpath, dirnames, filenames in os.walk(search_path):
            dirnames[:] = sorted([d for d in dirnames if d not in (".git", "node_modules")])
            for fname in sorted(filenames):
                if glob_pattern and not fnmatch.fnmatch(fname, glob_pattern):
                    continue
                full_path = os.path.join(dirpath, fname)
                try:
                    if not search_file(full_path):
                        break
                except Exception:
                    continue
            if match_limit_reached:
                break

    return {
        "matches": matches,
        "match_limit_reached": match_limit_reached,
        "lines_truncated": False,
        "is_directory": os.path.isdir(search_path),
    }


def create_grep_tool(cwd: str) -> GrepTool:
    return GrepTool(cwd)
