"""Minimal extension: adds a greeting tool and /hello command."""
from __future__ import annotations

from typing import Any, Dict, Optional

EXTENSION_NAME = "hello-example"


class HelloTool:
    name = "hello"
    label = "hello"
    description = "Return a short greeting."
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name to greet"},
        },
    }

    async def execute(
        self,
        tool_call_id: str,
        args: Dict[str, Any],
        signal: Optional[object] = None,
        on_update: Optional[object] = None,
    ) -> Dict[str, Any]:
        target = args.get("name", "there")
        return {"content": [{"type": "text", "text": f"Hello, {target}!"}]}


async def on_start(ctx):
    print(f"[{EXTENSION_NAME}] loaded in {ctx.cwd}")


def _hello_command(args: str, ctx):
    who = args.strip() or "there"
    return f"Hello, {who}!"


COMMANDS = [
    {
        "name": "hello",
        "description": "Print a greeting",
        "handler": _hello_command,
    }
]

# Expose the tool to the agent
custom_tools = [HelloTool()]
