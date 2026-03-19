"""Footer status bar component."""
from __future__ import annotations

from typing import Optional

from textual.widget import Widget
from rich.text import Text


class FooterWidget(Widget):
    """Status bar showing model, thinking level, token usage etc."""

    DEFAULT_CSS = """
    FooterWidget {
        height: 1;
        background: #2d2d2d;
        color: white;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_name: str = ""
        self._thinking_level: str = "off"
        self._tokens: int = 0
        self._cost: float = 0.0

    def update(
        self,
        model_name: Optional[str] = None,
        thinking_level: Optional[str] = None,
        tokens: Optional[int] = None,
        cost: Optional[float] = None,
    ) -> None:
        if model_name is not None:
            self._model_name = model_name
        if thinking_level is not None:
            self._thinking_level = thinking_level
        if tokens is not None:
            self._tokens = tokens
        if cost is not None:
            self._cost = cost
        self.refresh()

    def render(self):
        parts = []
        if self._model_name:
            parts.append(f"[bold]{self._model_name}[/bold]")
        if self._thinking_level and self._thinking_level != "off":
            parts.append(f"thinking:{self._thinking_level}")
        if self._tokens:
            parts.append(f"{self._tokens:,} tokens")
        if self._cost:
            parts.append(f"${self._cost:.4f}")
        parts.append("[dim]Ctrl+J send  Ctrl+C abort/quit[/dim]")
        return Text.from_markup("  |  ".join(parts) if parts else "Ready")
