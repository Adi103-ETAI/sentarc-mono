"""Theme configuration for interactive mode."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class InteractiveTheme:
    user_message: str = "blue"
    assistant_message: str = "green"
    tool_running: str = "yellow"
    tool_success: str = "green"
    tool_error: str = "red"
    thinking: str = "dim cyan"
    input_border: str = "blue"
    input_focused_border: str = "bright_blue"
    footer_bg: str = "#2d2d2d"
    footer_text: str = "white"
    accent: str = "cyan"
    muted: str = "dim"
    error: str = "red"
    warning: str = "yellow"
    success: str = "green"


DARK_THEME = InteractiveTheme()

LIGHT_THEME = InteractiveTheme(
    user_message="blue",
    assistant_message="dark_green",
    footer_bg="#e8e8e8",
    footer_text="black",
    thinking="italic dim",
    muted="dim",
)


def load_theme(name: str, themes_dir: Optional[Path] = None) -> InteractiveTheme:
    if name == "dark":
        return DARK_THEME
    if name == "light":
        return LIGHT_THEME
    if themes_dir:
        path = themes_dir / f"{name}.json"
        if path.exists():
            return _load_theme_from_file(path)
    return DARK_THEME


def _load_theme_from_file(path: Path) -> InteractiveTheme:
    data = json.loads(path.read_text())
    fields = InteractiveTheme.__dataclass_fields__
    return InteractiveTheme(**{k: v for k, v in data.items() if k in fields})
