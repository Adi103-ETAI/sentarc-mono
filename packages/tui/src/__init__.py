"""
sentarc-tui: Terminal UI library for Sentarc using textual.

Public API — import everything consumers need from here.
"""

from .types import Component, Focusable, OverlayOptions
from .tui import TUIApp
from .theme import Theme, DARK_THEME, LIGHT_THEME, load_theme, load_theme_from_file
from .utils import visible_width, strip_ansi, truncate_to_width
from .keys import (
    CTRL_C,
    CTRL_D,
    CTRL_Z,
    ENTER,
    ESC,
    BACKSPACE,
    TAB,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    matches_key,
)
from .components import (
    TextComponent,
    InputComponent,
    EditorComponent,
    MarkdownComponent,
    SelectListComponent,
    ContainerComponent,
    LoaderComponent,
)

__all__ = [
    # Types
    "Component",
    "Focusable",
    "OverlayOptions",
    # App
    "TUIApp",
    # Theme
    "Theme",
    "DARK_THEME",
    "LIGHT_THEME",
    "load_theme",
    "load_theme_from_file",
    # Utils
    "visible_width",
    "strip_ansi",
    "truncate_to_width",
    # Keys
    "CTRL_C",
    "CTRL_D",
    "CTRL_Z",
    "ENTER",
    "ESC",
    "BACKSPACE",
    "TAB",
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "matches_key",
    # Components
    "TextComponent",
    "InputComponent",
    "EditorComponent",
    "MarkdownComponent",
    "SelectListComponent",
    "ContainerComponent",
    "LoaderComponent",
]
