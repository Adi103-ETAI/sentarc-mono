"""
Single-line input component with history and callbacks.
"""
from __future__ import annotations

from collections import deque
from typing import Callable

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import Input
from textual.message import Message


class InputComponent(Input):
    """
    Single-line text input with submit/change callbacks and command history.

    Parameters
    ----------
    placeholder:
        Placeholder text shown when the field is empty.
    on_change:
        Called with the current value on every keystroke.
    on_submit:
        Called with the final value when the user presses Enter.
    history_size:
        Maximum number of submitted values to remember (navigate with
        Up/Down arrows while the field is empty or matches history).
    """

    DEFAULT_CSS = """
    InputComponent {
        border: tall $accent;
        margin: 0 0 1 0;
    }
    """

    def __init__(
        self,
        placeholder: str = "",
        *,
        on_change: Callable[[str], None] | None = None,
        on_submit: Callable[[str], None] | None = None,
        history_size: int = 100,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(placeholder=placeholder, id=id, classes=classes)
        self._on_change_cb = on_change
        self._on_submit_cb = on_submit
        self._history: deque[str] = deque(maxlen=history_size)
        self._history_index: int = -1   # -1 → current (not navigating)
        self._saved_value: str = ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_value(self) -> str:
        """The current text value of the input field."""
        return self.value

    # ------------------------------------------------------------------
    # Textual event handlers
    # ------------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        event.stop()
        if self._on_change_cb is not None:
            self._on_change_cb(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        value = event.value.strip()
        if value:
            self._history.appendleft(value)
        self._history_index = -1
        self._saved_value = ""
        if self._on_submit_cb is not None:
            self._on_submit_cb(value)
        self.clear()

    def on_key(self, event) -> None:
        if event.key == "up":
            self._navigate_history(-1)
            event.prevent_default()
        elif event.key == "down":
            self._navigate_history(1)
            event.prevent_default()

    # ------------------------------------------------------------------
    # History navigation
    # ------------------------------------------------------------------

    def _navigate_history(self, direction: int) -> None:
        """direction: -1 = older, +1 = newer."""
        if not self._history:
            return

        if self._history_index == -1 and direction == -1:
            # Start navigating — save the current draft
            self._saved_value = self.value
            self._history_index = 0
        elif direction == 1 and self._history_index <= 0:
            # Return to the draft
            self._history_index = -1
            self.value = self._saved_value
            return
        else:
            new_index = self._history_index + direction
            if new_index < 0:
                new_index = 0
            elif new_index >= len(self._history):
                return
            self._history_index = new_index

        if self._history_index >= 0:
            self.value = self._history[self._history_index]

    # ------------------------------------------------------------------
    # Component protocol shim
    # ------------------------------------------------------------------

    def render(self, width: int = 0) -> list[str]:  # type: ignore[override]
        return [self.value]

    def handle_input(self, data: str) -> None:
        pass

    def invalidate(self) -> None:
        self.refresh()
