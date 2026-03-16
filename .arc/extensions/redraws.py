"""redraws — /tui command shows TUI redraw statistics."""

from typing import Any

EXTENSION_NAME = "redraws"


async def _tui_handler(args: str, ctx: Any) -> str:
    """Show TUI redraw statistics."""
    return "TUI stats: use in interactive mode for redraw metrics."


COMMANDS = [
    {
        "name": "tui",
        "description": "Show TUI stats",
        "handler": _tui_handler,
    },
]
