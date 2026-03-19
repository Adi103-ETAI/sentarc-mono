"""
Static text component.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Static
from textual.reactive import reactive


class TextComponent(Static):
    """
    A static text display component.

    Wraps textual's :class:`~textual.widgets.Static` and adds:
    - A :attr:`text` reactive so the content can be updated at any time.
    - ``render()`` compatibility shim so it also satisfies the
      :class:`~sentarc_tui.types.Component` protocol.
    """

    text: reactive[str] = reactive("", layout=True)

    def __init__(
        self,
        text: str = "",
        *,
        markup: bool = True,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(text, markup=markup, id=id, classes=classes)
        self.text = text

    # ------------------------------------------------------------------
    # Reactive watcher
    # ------------------------------------------------------------------

    def watch_text(self, new_text: str) -> None:
        self.update(new_text)

    # ------------------------------------------------------------------
    # Component protocol shim
    # ------------------------------------------------------------------

    def render(self, width: int = 0) -> list[str]:  # type: ignore[override]
        """Return the text split into lines (protocol shim, not used by textual)."""
        return self.text.splitlines() or [""]

    def handle_input(self, data: str) -> None:
        pass

    def invalidate(self) -> None:
        self.refresh()
