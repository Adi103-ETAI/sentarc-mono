"""Assistant message display with streaming support."""
from __future__ import annotations

from textual.widget import Widget
from textual.reactive import reactive
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text


class AssistantMessageWidget(Widget):
    """Displays streaming assistant message with markdown rendering."""

    text: reactive[str] = reactive("", layout=True)
    is_streaming: reactive[bool] = reactive(False)

    DEFAULT_CSS = """
    AssistantMessageWidget {
        margin: 0 0 1 0;
        height: auto;
    }
    """

    def render(self):
        status = " ●" if self.is_streaming else ""
        content = Markdown(self.text) if self.text else Text("")
        return Panel(
            content,
            title=f"[bold green]Assistant[/bold green]{status}",
            border_style="green",
            padding=(0, 1),
        )

    def append_text(self, delta: str) -> None:
        self.text += delta

    def set_streaming(self, streaming: bool) -> None:
        self.is_streaming = streaming
