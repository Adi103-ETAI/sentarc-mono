"""
Layout container component.
"""
from __future__ import annotations

from typing import Literal

from textual.app import ComposeResult
from textual.widget import Widget
from textual.containers import (
    Container,
    Horizontal,
    Vertical,
    ScrollableContainer,
)


_LAYOUT_MAP = {
    "vertical": Vertical,
    "horizontal": Horizontal,
    "scroll": ScrollableContainer,
    "default": Container,
}

Layout = Literal["vertical", "horizontal", "scroll", "default"]


class ContainerComponent(Widget):
    """
    A flexible layout container.

    Parameters
    ----------
    layout:
        ``"vertical"`` (default), ``"horizontal"``, ``"scroll"``, or
        ``"default"`` (CSS-controlled).
    children:
        Child widgets to mount immediately.
    """

    DEFAULT_CSS = """
    ContainerComponent {
        height: auto;
        width: 1fr;
    }
    """

    def __init__(
        self,
        *children: Widget,
        layout: Layout = "vertical",
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self._layout = layout
        self._initial_children = list(children)

    def compose(self) -> ComposeResult:
        container_cls = _LAYOUT_MAP.get(self._layout, Container)
        with container_cls():
            yield from self._initial_children

    # ------------------------------------------------------------------
    # Component protocol shim
    # ------------------------------------------------------------------

    def render(self, width: int = 0) -> list[str]:  # type: ignore[override]
        return [f"<ContainerComponent layout={self._layout!r}>"]

    def handle_input(self, data: str) -> None:
        pass

    def invalidate(self) -> None:
        self.refresh()
