"""Example slash commands extension."""
from __future__ import annotations

from datetime import datetime
from typing import Any

EXTENSION_NAME = "commands-example"


def _echo(args: str, ctx: Any) -> str:
    return f"echo: {args}" if args else "echo: <empty>"


def _time(_: str, ctx: Any) -> str:
    return datetime.now().isoformat(timespec="seconds")


def _branch_info(_: str, ctx: Any) -> str:
    if not ctx.session_manager:
        return "no session"
    leaf = ctx.session_manager.get_leaf_entry()
    return f"leaf: {leaf.get('id')} type={leaf.get('type')}" if leaf else "empty session"


COMMANDS = [
    {"name": "echo", "description": "Echo the provided text", "handler": _echo},
    {"name": "time", "description": "Show current time", "handler": _time},
    {"name": "branch-info", "description": "Show current branch leaf", "handler": _branch_info},
]
