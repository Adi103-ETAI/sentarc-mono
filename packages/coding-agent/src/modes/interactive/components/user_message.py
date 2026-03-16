"""User message display component."""
from __future__ import annotations

from textual.widget import Widget
from rich.panel import Panel
from rich.text import Text


class UserMessageWidget(Widget):
    """Displays a user message in a styled panel."""

    DEFAULT_CSS = """
    UserMessageWidget {
        margin: 0 0 1 0;
        height: auto;
    }
    """

    def __init__(self, text: str, **kwargs):
        super().__init__(**kwargs)
        self._text = text

    def render(self):
        return Panel(
            Text(self._text),
            title="[bold blue]You[/bold blue]",
            border_style="blue",
            padding=(0, 1),
        )
