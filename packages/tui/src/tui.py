"""
TUI orchestrator — thin wrapper around textual App providing Component lifecycle.
"""
from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer
from textual.screen import Screen
from textual.containers import Container

from .types import Component, OverlayOptions


class _OverlayScreen(Screen):
    """A simple full-screen overlay that wraps a textual widget."""

    DEFAULT_CSS = """
    _OverlayScreen {
        align: center middle;
    }
    _OverlayScreen > Container {
        width: 80%;
        height: 80%;
        border: round $accent;
        padding: 1 2;
    }
    """

    def __init__(self, widget, title: str = "", closeable: bool = True) -> None:
        super().__init__()
        self._widget = widget
        self._title = title
        self._closeable = closeable

    def compose(self) -> ComposeResult:
        with Container():
            if self._title:
                from textual.widgets import Label
                yield Label(self._title, id="overlay-title")
            yield self._widget

    def on_key(self, event) -> None:
        if self._closeable and event.key == "escape":
            self.dismiss()


class TUIApp(App):
    """
    Main TUI application wrapping textual App with Component management.

    Subclass this and override `compose()` to build your UI, or use
    `add_child()` / `show_overlay()` at runtime.
    """

    CSS = """
    TUIApp {
        layers: base overlay;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._children: list[Component] = []
        self._overlay: Component | None = None

    # ------------------------------------------------------------------
    # Component management
    # ------------------------------------------------------------------

    def add_child(self, component: Component) -> None:
        """Register a component and mount it if the app is already running."""
        self._children.append(component)
        if hasattr(component, "mount") and self.is_running:
            # component is a textual widget — mount it
            self.mount(component)  # type: ignore[arg-type]

    def remove_child(self, component: Component) -> None:
        """Unregister and unmount a component."""
        if component in self._children:
            self._children.remove(component)
        if hasattr(component, "remove"):
            component.remove()  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Overlay management
    # ------------------------------------------------------------------

    def show_overlay(
        self,
        component: Component,
        options: OverlayOptions | None = None,
    ) -> None:
        """Push an overlay screen containing `component`."""
        opts = options or OverlayOptions()
        self._overlay = component
        widget = component if hasattr(component, "compose") else None
        if widget is None:
            raise TypeError(
                "show_overlay requires a textual Widget as component"
            )
        screen = _OverlayScreen(
            widget=widget,
            title=opts.title,
            closeable=opts.closeable,
        )
        self.push_screen(screen)

    def hide_overlay(self) -> None:
        """Pop the topmost overlay screen."""
        self._overlay = None
        if len(self.screen_stack) > 1:
            self.pop_screen()

    def has_overlay(self) -> bool:
        """Return True if an overlay is currently visible."""
        return self._overlay is not None

    # ------------------------------------------------------------------
    # Render helpers
    # ------------------------------------------------------------------

    def request_render(self) -> None:
        """Request a full screen refresh."""
        self.refresh()
