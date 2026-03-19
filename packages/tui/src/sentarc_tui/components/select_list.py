"""
Keyboard-navigable selection list component.
"""
from __future__ import annotations

from typing import Callable, Generic, TypeVar

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import ListView, ListItem, Label
from textual.message import Message

T = TypeVar("T")


class SelectListComponent(ListView, Generic[T]):
    """
    A keyboard-navigable selection list.

    Displays a list of items (converted to strings via ``str()``), lets
    the user move the highlight with Up/Down and confirm with Enter.

    Parameters
    ----------
    items:
        The items to display.
    on_select:
        Called with ``(index, item)`` when the user presses Enter.
    selected_index:
        Initially highlighted index (default 0).
    """

    DEFAULT_CSS = """
    SelectListComponent {
        border: tall $accent;
        height: auto;
        max-height: 20;
        margin: 0 0 1 0;
    }
    SelectListComponent > ListItem.--highlight {
        background: $accent;
        color: $text;
    }
    """

    def __init__(
        self,
        items: list[T] | None = None,
        *,
        on_select: Callable[[int, T], None] | None = None,
        selected_index: int = 0,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        self._items: list[T] = list(items or [])
        self._on_select_cb = on_select
        list_items = [ListItem(Label(str(item))) for item in self._items]
        super().__init__(*list_items, id=id, classes=classes, initial_index=selected_index)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def items(self) -> list[T]:
        """The current list of items."""
        return list(self._items)

    @property
    def selected_index(self) -> int:
        """Index of the currently highlighted item."""
        return self.index if self.index is not None else -1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_items(self, items: list[T]) -> None:
        """Replace the list contents and re-render."""
        self._items = list(items)
        self.clear()
        for item in self._items:
            self.append(ListItem(Label(str(item))))

    def get_selected(self) -> T | None:
        """Return the currently highlighted item, or None if the list is empty."""
        idx = self.selected_index
        if 0 <= idx < len(self._items):
            return self._items[idx]
        return None

    # ------------------------------------------------------------------
    # Textual event handlers
    # ------------------------------------------------------------------

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        event.stop()
        idx = self.selected_index
        if self._on_select_cb is not None and 0 <= idx < len(self._items):
            self._on_select_cb(idx, self._items[idx])

    # ------------------------------------------------------------------
    # Component protocol shim
    # ------------------------------------------------------------------

    def render(self, width: int = 0) -> list[str]:  # type: ignore[override]
        idx = self.selected_index
        lines: list[str] = []
        for i, item in enumerate(self._items):
            prefix = "> " if i == idx else "  "
            lines.append(f"{prefix}{item}")
        return lines or ["(empty)"]

    def handle_input(self, data: str) -> None:
        pass

    def invalidate(self) -> None:
        self.refresh()
