"""Find tool — glob file search respecting .gitignore."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from sentarc_coding_agent.core.tools.path_utils import resolve_to_cwd
from sentarc_coding_agent.core.tools.truncate import (
    DEFAULT_MAX_BYTES,
    TruncationOptions,
    format_size,
    truncate_head,
)

DEFAULT_LIMIT = 1000


def _load_gitignore_spec(root: str):
    """Load a pathspec matcher from .gitignore files, if pathspec is available."""
    try:
        import pathspec
    except ImportError:
        return None

    patterns: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip .git and node_modules
        dirnames[:] = [d for d in dirnames if d not in (".git", "node_modules")]
        for name in (".gitignore", ".ignore"):
            gi_path = os.path.join(dirpath, name)
            if os.path.isfile(gi_path):
                try:
                    with open(gi_path, "r", encoding="utf-8", errors="replace") as f:
                        rel_dir = os.path.relpath(dirpath, root)
                        prefix = "" if rel_dir == "." else rel_dir.replace(os.sep, "/") + "/"
                        for line in f:
                            line = line.rstrip("\n")
                            stripped = line.strip()
                            if not stripped or stripped.startswith("#"):
                                continue
                            if stripped.startswith("!"):
                                patterns.append("!" + prefix + stripped[1:].lstrip("/"))
                            elif stripped.startswith("/"):
                                patterns.append(prefix + stripped[1:])
                            else:
                                patterns.append(prefix + stripped)
                except Exception:
                    pass

    if not patterns:
        return None
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def _find_files_glob(pattern: str, root: str, limit: int, spec) -> List[str]:
    """Find files matching pattern under root, respecting gitignore spec."""
    import fnmatch

    results: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted([d for d in dirnames if d not in (".git", "node_modules")])
        all_entries = sorted(filenames) + sorted([d + "/" for d in dirnames])
        for name in all_entries:
            is_dir = name.endswith("/")
            bare_name = name.rstrip("/")
            full_path = os.path.join(dirpath, bare_name)
            rel_path = os.path.relpath(full_path, root).replace(os.sep, "/")
            if is_dir:
                rel_path_check = rel_path + "/"
            else:
                rel_path_check = rel_path

            if spec and spec.match_file(rel_path_check):
                if is_dir:
                    dirnames[:] = [d for d in dirnames if d != bare_name]
                continue

            if not is_dir and fnmatch.fnmatch(bare_name, pattern):
                results.append(rel_path)
                if len(results) >= limit:
                    return results

    return results


class FindTool:
    """AgentTool for finding files by glob pattern."""

    name = "find"
    label = "find"
    description = (
        f"Search for files by glob pattern. Returns matching file paths relative to "
        f"the search directory. Respects .gitignore. Output is truncated to "
        f"{DEFAULT_LIMIT} results or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match files, e.g. '*.ts', '**/*.json'",
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: current directory)",
            },
            "limit": {
                "type": "integer",
                "description": f"Maximum number of results (default: {DEFAULT_LIMIT})",
            },
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
        limit: int = args.get("limit") or DEFAULT_LIMIT

        if signal and signal.is_set():
            raise Exception("Operation aborted")

        search_path = resolve_to_cwd(search_dir or ".", self.cwd)

        if not os.path.exists(search_path):
            raise Exception(f"Path not found: {search_path}")

        # Try using fd (fast-find) first
        fd_path = _find_fd()
        if fd_path:
            results = await _run_fd(fd_path, pattern, search_path, limit, signal)
        else:
            spec = _load_gitignore_spec(search_path)
            results = _find_files_glob(pattern, search_path, limit, spec)

        if not results:
            return {"content": [{"type": "text", "text": "No files found matching pattern"}], "details": None}

        result_limit_reached = len(results) >= limit
        raw_output = "\n".join(results)
        truncation = truncate_head(raw_output, TruncationOptions(max_lines=2**31 - 1))

        output = truncation.content
        details: Dict[str, Any] = {}
        notices: List[str] = []

        if result_limit_reached:
            notices.append(
                f"{limit} results limit reached. Use limit={limit * 2} for more, or refine pattern"
            )
            details["resultLimitReached"] = limit

        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details["truncation"] = truncation

        if notices:
            output += f"\n\n[{'. '.join(notices)}]"

        return {
            "content": [{"type": "text", "text": output}],
            "details": details if details else None,
        }


def _find_fd() -> Optional[str]:
    """Find the 'fd' binary."""
    import shutil
    return shutil.which("fd") or shutil.which("fdfind")


async def _run_fd(
    fd_path: str,
    pattern: str,
    search_path: str,
    limit: int,
    signal: Optional[asyncio.Event],
) -> List[str]:
    """Run fd to find files."""
    args = [
        "--glob",
        "--color=never",
        "--hidden",
        "--max-results", str(limit),
        pattern,
        search_path,
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            fd_path,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30.0)
    except Exception:
        return []

    output = stdout.decode("utf-8", errors="replace").strip()
    if not output:
        return []

    results = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(search_path):
            rel = line[len(search_path):].lstrip("/").lstrip(os.sep)
        else:
            rel = os.path.relpath(line, search_path)
        results.append(rel.replace(os.sep, "/"))

    return results


def create_find_tool(cwd: str) -> FindTool:
    return FindTool(cwd)
