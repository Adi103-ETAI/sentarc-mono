"""Blocks dangerous bash commands before they run."""
from __future__ import annotations

DANGEROUS = ("rm -rf", ":(){", "mkfs", "shutdown", "reboot")

EXTENSION_NAME = "permission-gate"


def on_tool_call(ctx, tool_name: str, args):
    if tool_name != "bash":
        return None
    cmd = (args or {}).get("command", "")
    if any(token in cmd for token in DANGEROUS):
        raise RuntimeError(f"blocked dangerous bash command: {cmd}")
    return None
