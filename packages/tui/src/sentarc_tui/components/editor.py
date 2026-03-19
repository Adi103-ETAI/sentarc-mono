"""
Multi-line editor component with word-wrap and Ctrl+Enter submit.
"""
from __future__ import annotations

from typing import Callable

from textual.widgets import TextArea
from textual.reactive import reactive


class EditorComponent(TextArea):
    """
    Multi-line text editor with word-wrap and submit/change callbacks.

    Parameters
    ----------
    initial_text:
        Initial content of the editor.
    on_change:
        Called with the full text on every edit.
    on_submit:
        Called with the full text when the user presses Ctrl+Enter.
    language:
        Optional syntax-highlighting language (e.g. ``"python"``).
    """

    DEFAULT_CSS = """
    EditorComponent {
        border: tall $accent;
        height: 10;
        margin: 0 0 1 0;
    }
    """

    def __init__(
        self,
        initial_text: str = "",
        *,
        on_change: Callable[[str], None] | None = None,
        on_submit: Callable[[str], None] | None = None,
        language: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(
            text=initial_text,
            language=language,
            id=id,
            classes=classes,
            soft_wrap=True,
        )
        self._on_change_cb = on_change
        self._on_submit_cb = on_submit

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_value(self) -> str:
        """The full text currently in the editor."""
        return self.text

    @current_value.setter
    def current_value(self, value: str) -> None:
        self.load_text(value)

    # ------------------------------------------------------------------
    # Textual event handlers
    # ------------------------------------------------------------------

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        event.stop()
        if self._on_change_cb is not None:
            self._on_change_cb(self.text)

    def on_key(self, event) -> None:
        # Ctrl+Enter → submit
        if event.key == "ctrl+j" or (
            event.character == "\n" and event.ctrl
        ):
            event.prevent_default()
            if self._on_submit_cb is not None:
                self._on_submit_cb(self.text)
            return

    # ------------------------------------------------------------------
    # Component protocol shim
    # ------------------------------------------------------------------

    def render(self, width: int = 0) -> list[str]:  # type: ignore[override]
        return self.text.splitlines() or [""]

    def handle_input(self, data: str) -> None:
        pass

    def invalidate(self) -> None:
        self.refresh()
