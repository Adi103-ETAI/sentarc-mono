"""
Animated loading spinner component.
"""
from __future__ import annotations

from textual.widgets import LoadingIndicator


class LoaderComponent(LoadingIndicator):
    """
    Animated loading spinner built on textual's
    :class:`~textual.widgets.LoadingIndicator`.

    Parameters
    ----------
    label:
        Optional label shown alongside the spinner.  When empty only the
        spinner is rendered.
    """

    DEFAULT_CSS = """
    LoaderComponent {
        height: 3;
        margin: 0 0 1 0;
    }
    """

    def __init__(
        self,
        label: str = "",
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self._label = label

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        self._label = value
        self.refresh()

    def start(self) -> None:
        """Show the spinner (remove the ``hidden`` CSS class if set)."""
        self.remove_class("hidden")

    def stop(self) -> None:
        """Hide the spinner by adding the ``hidden`` CSS class."""
        self.add_class("hidden")

    # ------------------------------------------------------------------
    # Component protocol shim
    # ------------------------------------------------------------------

    def render(self, width: int = 0) -> list[str]:  # type: ignore[override]
        return [f"[loading] {self._label}".strip()]

    def handle_input(self, data: str) -> None:
        pass

    def invalidate(self) -> None:
        self.refresh()
