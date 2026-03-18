"""Blocks edits/writes to sensitive paths."""
from __future__ import annotations

import os
from typing import Dict

EXTENSION_NAME = "protected-paths"

PROTECTED_ROOTS = ["/etc", "/var", "/usr", "~/.ssh", "~/.config"]


def _is_protected(path: str, cwd: str) -> bool:
    abs_path = os.path.abspath(os.path.expanduser(os.path.join(cwd, path)))
    for root in PROTECTED_ROOTS:
        root_abs = os.path.abspath(os.path.expanduser(root))
        if os.path.commonpath([abs_path, root_abs]) == root_abs:
            return True
    return False


def on_tool_call(ctx, tool_name: str, args: Dict):
    if tool_name not in ("write", "edit"):
        return None
    path = (args or {}).get("path")
    if not path:
        return None
    if _is_protected(path, ctx.cwd):
        raise RuntimeError(f"writes to protected path blocked: {path}")
    return None
