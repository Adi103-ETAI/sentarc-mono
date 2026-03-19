"""Theme dataclass and built-in dark/light presets for arc TUI components."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Theme:
    """Colour and style tokens used throughout the TUI."""

    # Message borders / highlights
    user_message_border: str = "blue"
    assistant_message_border: str = "green"
    tool_border: str = "yellow"
    error_color: str = "red"
    thinking_color: str = "dim"

    # Footer bar
    footer_bg: str = "#333333"
    footer_fg: str = "white"

    # General accents
    accent: str = "cyan"
    muted: str = "dim white"

    # Input / editor
    input_border: str = "blue"
    input_placeholder: str = "dim"

    # Selection list
    selected_bg: str = "blue"
    selected_fg: str = "white"


# ---------------------------------------------------------------------------
# Built-in themes
# ---------------------------------------------------------------------------

DARK_THEME = Theme()

LIGHT_THEME = Theme(
    user_message_border="blue",
    assistant_message_border="green",
    tool_border="dark_orange",
    error_color="red",
    thinking_color="grey50",
    footer_bg="#dddddd",
    footer_fg="black",
    accent="dark_cyan",
    muted="grey50",
    input_border="blue",
    input_placeholder="grey50",
    selected_bg="steel_blue1",
    selected_fg="white",
)

_REGISTRY: dict[str, Theme] = {
    "dark": DARK_THEME,
    "light": LIGHT_THEME,
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_theme(name: str) -> Theme:
    """
    Return a built-in theme by name (``"dark"`` or ``"light"``).

    Raises :class:`KeyError` for unknown names.
    """
    try:
        return _REGISTRY[name.lower()]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown theme {name!r}. Available: {available}") from None


def load_theme_from_file(path: str) -> Theme:
    """
    Load a :class:`Theme` from a JSON file.

    The JSON file may contain a subset of the :class:`Theme` fields;
    missing fields fall back to their defaults.

    Example JSON::

        {
            "accent": "magenta",
            "user_message_border": "cyan"
        }
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Theme file {path!r} must contain a JSON object.")
    return Theme(**{k: v for k, v in data.items() if k in Theme.__dataclass_fields__})
