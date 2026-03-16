# sentarc-tui

Terminal UI library for Sentarc — textual-based components and TUI orchestration.

## Overview

`sentarc-tui` provides reusable terminal UI components and a TUI orchestrator built on top of [textual](https://textual.textualize.io/), used by `sentarc-coding-agent` and other consumers.

## Components

| Component | Description |
|-----------|-------------|
| `TextComponent` | Static text display |
| `InputComponent` | Single-line text input with history |
| `EditorComponent` | Multi-line editor (Ctrl+Enter to submit) |
| `MarkdownComponent` | Rich Markdown renderer |
| `SelectListComponent` | Keyboard-navigable selection list |
| `ContainerComponent` | Layout container |
| `LoaderComponent` | Animated loading spinner |

## Usage

```python
from sentarc_tui import TUIApp, InputComponent, MarkdownComponent, DARK_THEME

class MyApp(TUIApp):
    def compose(self):
        yield InputComponent(on_submit=self.handle_input)

    def handle_input(self, value: str) -> None:
        self.notify(f"You typed: {value}")

MyApp().run()
```

## Theming

```python
from sentarc_tui import DARK_THEME, LIGHT_THEME, load_theme

theme = load_theme("dark")
```

## Installation

```bash
pip install -e packages/tui
```
