"""Tool execution display component."""
from __future__ import annotations

from typing import Any, Dict, Optional

from textual.widget import Widget
from textual.reactive import reactive
from rich.panel import Panel
from rich.text import Text


class ToolExecutionWidget(Widget):
    """Shows tool call with name, args, and result."""

    status: reactive[str] = reactive("running")  # running | success | error

    DEFAULT_CSS = """
    ToolExecutionWidget {
        margin: 0 0 1 0;
        height: auto;
    }
    """

    def __init__(self, tool_name: str, args: Optional[Dict[str, Any]], **kwargs):
        super().__init__(**kwargs)
        self._tool_name = tool_name
        self._args = args or {}
        self._result = None

    def set_result(self, result: Any, is_error: bool = False) -> None:
        self._result = result
        self.status = "error" if is_error else "success"

    def render(self):
        if self.status == "running":
            title = f"[yellow]⟳ {self._tool_name}[/yellow]"
            border = "yellow"
        elif self.status == "error":
            title = f"[red]✗ {self._tool_name}[/red]"
            border = "red"
        else:
            title = f"[green]✓ {self._tool_name}[/green]"
            border = "green"

        args_str = ", ".join(f"{k}={repr(v)[:40]}" for k, v in self._args.items())
        body = Text(f"({args_str})", style="dim")
        return Panel(body, title=title, border_style=border, padding=(0, 1))
