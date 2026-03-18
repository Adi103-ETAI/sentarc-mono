# sentarc-tui

Terminal UI library for Sentarc built on [Textual](https://textual.textualize.io/) — reusable components and TUI orchestration.

## Features

- **Component-based**: Simple Component interface with `render()` method
- **Built on Textual**: Leverages Textual's reactive system and rendering
- **Theme Support**: Customizable color themes with dark/light presets
- **Built-in Components**: Text, Input, Editor, Markdown, SelectList, Container, Loader
- **Overlay Support**: Modal dialogs and overlays with focus management
- **Unicode Support**: Proper handling of CJK double-width characters

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core API](#core-api)
  - [TUIApp](#tuiapp)
  - [Overlays](#overlays)
  - [Component Interface](#component-interface)
- [Built-in Components](#built-in-components)
- [Theming](#theming)
- [Utilities](#utilities)
- [Key Detection](#key-detection)
- [Creating Custom Components](#creating-custom-components)

---

## Installation

Install from PyPI:

```bash
pip install sentarc-tui
```

Or for development:

```bash
pip install -e packages/tui
```

---

## Quick Start

```python
from sentarc_tui import TUIApp, TextComponent, InputComponent, DARK_THEME

class MyApp(TUIApp):
    def compose(self):
        yield TextComponent("Welcome to my app!")
        yield InputComponent(
            placeholder="Type something...",
            on_submit=self.handle_input
        )

    def handle_input(self, value: str) -> None:
        self.add_child(TextComponent(f"You said: {value}"))

if __name__ == "__main__":
    MyApp().run()
```

---

## Core API

### TUIApp

Main application class that manages components and rendering.

```python
from sentarc_tui import TUIApp

class MyApp(TUIApp):
    def compose(self):
        yield TextComponent("Hello!")

app = MyApp()
app.add_child(component)      # Add component dynamically
app.remove_child(component)   # Remove component
app.request_render()          # Request a re-render
app.run()                     # Start the application
```

### Overlays

Overlays render components on top of existing content. Useful for dialogs and modal UI.

```python
from sentarc_tui import OverlayOptions

# Show overlay with default options
handle = app.show_overlay(component)

# Show overlay with title
handle = app.show_overlay(component, OverlayOptions(
    title="My Dialog",
    closeable=True  # Allow ESC to close
))

# Hide overlay
app.hide_overlay()

# Check if any overlay is active
app.has_overlay()
```

### Component Interface

All components implement the `Component` protocol:

```python
from sentarc_tui import Component

class MyComponent(Component):
    def render(self, width: int) -> list[str]:
        """Return lines to display. Each line must not exceed width."""
        return ["Hello, World!"]
```

| Method | Description |
|--------|-------------|
| `render(width)` | Returns a list of strings, one per line. Each line must not exceed `width`. |

---

## Built-in Components

### TextComponent

Displays static text with optional markup.

```python
from sentarc_tui import TextComponent

text = TextComponent("Hello **World**!", markup=True)
text.text = "Updated text"  # Update content reactively
```

### InputComponent

Single-line text input with history support.

```python
from sentarc_tui import InputComponent

input = InputComponent(
    placeholder="Enter text...",
    on_change=lambda value: print(f"Changed: {value}"),
    on_submit=lambda value: print(f"Submitted: {value}"),
    history_size=100  # Command history (Up/Down arrows)
)

value = input.current_value  # Get current value
```

**Key Bindings:**
- `Enter` - Submit
- `Up` / `Down` - Navigate history
- Standard editing keys (Backspace, Delete, etc.)

### EditorComponent

Multi-line text editor with syntax highlighting.

```python
from sentarc_tui import EditorComponent

editor = EditorComponent(
    initial_text="",
    on_change=lambda text: print(f"Changed: {text}"),
    on_submit=lambda text: print(f"Submitted: {text}"),
    language="python"  # Optional syntax highlighting
)

editor.current_value = "new text"  # Set content
text = editor.current_value        # Get content
```

**Key Bindings:**
- `Ctrl+Enter` or `Ctrl+J` - Submit
- Standard editing keys
- Soft word wrapping enabled

### MarkdownComponent

Rich Markdown rendering.

```python
from sentarc_tui import MarkdownComponent

md = MarkdownComponent("# Hello\n\nSome **bold** text")
md.set_markdown_sync("# Updated")  # Sync update
await md.set_markdown("# Updated") # Async update
```

**Supported Features:**
- Headings, bold, italic, code blocks
- Lists, links, blockquotes
- Syntax highlighting in code blocks

### SelectListComponent

Interactive selection list with keyboard navigation.

```python
from sentarc_tui import SelectListComponent

items = ["Option 1", "Option 2", "Option 3"]
select = SelectListComponent(
    items=items,
    on_select=lambda index, item: print(f"Selected: {item}"),
    selected_index=0
)

select.set_items(new_items)       # Replace items
selected = select.get_selected()  # Get highlighted item
index = select.selected_index     # Get highlighted index
```

**Key Bindings:**
- `Up` / `Down` - Navigate
- `Enter` - Select

### ContainerComponent

Flexible layout container for grouping components.

```python
from sentarc_tui import ContainerComponent

# Vertical layout (default)
container = ContainerComponent(child1, child2, layout="vertical")

# Horizontal layout
container = ContainerComponent(child1, child2, layout="horizontal")

# Scrollable
container = ContainerComponent(child1, child2, layout="scroll")
```

**Layout Options:**
- `"vertical"` - Stack children vertically (default)
- `"horizontal"` - Stack children horizontally
- `"scroll"` - Scrollable container
- `"default"` - CSS-controlled layout

### LoaderComponent

Animated loading spinner.

```python
from sentarc_tui import LoaderComponent

loader = LoaderComponent(label="Loading...")
loader.start()                    # Show spinner
loader.label = "Still loading..." # Update label
loader.stop()                     # Hide spinner
```

---

## Theming

Built-in themes: `DARK_THEME`, `LIGHT_THEME`

```python
from sentarc_tui import Theme, DARK_THEME, LIGHT_THEME, load_theme, load_theme_from_file

# Use built-in theme
theme = DARK_THEME

# Load by name
theme = load_theme("dark")  # or "light"

# Load from JSON file (partial fields supported)
theme = load_theme_from_file("my_theme.json")
```

### Theme Tokens

| Token | Description |
|-------|-------------|
| `user_message_border` | User message border color |
| `assistant_message_border` | Assistant message border color |
| `tool_border` | Tool output border color |
| `error_color` | Error text color |
| `thinking_color` | Thinking/processing text color |
| `footer_bg` | Footer background color |
| `footer_fg` | Footer text color |
| `accent` | Accent color for highlights |
| `muted` | Muted/secondary text color |
| `input_border` | Input field border color |
| `input_placeholder` | Input placeholder color |
| `selected_bg` | Selected item background |
| `selected_fg` | Selected item text color |

### Custom Theme JSON

```json
{
  "user_message_border": "blue",
  "assistant_message_border": "green",
  "accent": "cyan",
  "footer_bg": "#333333",
  "footer_fg": "white"
}
```

---

## Utilities

```python
from sentarc_tui import visible_width, strip_ansi, truncate_to_width

# Get visible width of string (ignoring ANSI codes, handling CJK)
width = visible_width("\x1b[31mHello\x1b[0m")  # 5
width = visible_width("你好")                   # 4 (CJK double-width)

# Remove all ANSI escape sequences
plain = strip_ansi("\x1b[31mHello\x1b[0m")  # "Hello"

# Truncate string to width (preserving ANSI codes)
truncated = truncate_to_width("Hello World", 8)  # "Hello..."
```

---

## Key Detection

Use `matches_key()` to detect keyboard input:

```python
from sentarc_tui import matches_key, CTRL_C, ENTER, ESC, UP, DOWN

def handle_input(data: str) -> None:
    if matches_key(data, CTRL_C):
        exit()
    elif matches_key(data, ENTER):
        submit()
    elif matches_key(data, ESC):
        cancel()
    elif matches_key(data, UP):
        move_up()
    elif matches_key(data, DOWN):
        move_down()
```

**Available Keys:**

| Category | Keys |
|----------|------|
| Control | `CTRL_A` - `CTRL_Z`, `CTRL_C`, `CTRL_D`, `CTRL_L`, `CTRL_Z` |
| Common | `ENTER`, `ESC`, `BACKSPACE`, `TAB`, `DELETE` |
| Navigation | `UP`, `DOWN`, `LEFT`, `RIGHT`, `HOME`, `END`, `PAGE_UP`, `PAGE_DOWN` |

---

## Creating Custom Components

### Basic Component

```python
from sentarc_tui import Component, truncate_to_width

class MyComponent(Component):
    def __init__(self, text: str):
        self.text = text

    def render(self, width: int) -> list[str]:
        # Each line must not exceed width
        return [truncate_to_width(self.text, width)]
```

### Interactive Component

```python
from sentarc_tui import Component, Focusable, matches_key, UP, DOWN, ENTER, ESC

class MySelector(Component, Focusable):
    focused: bool = False

    def __init__(self, items: list[str]):
        self.items = items
        self.selected = 0
        self.on_select = None
        self.on_cancel = None

    def handle_input(self, data: str) -> None:
        if matches_key(data, UP):
            self.selected = max(0, self.selected - 1)
        elif matches_key(data, DOWN):
            self.selected = min(len(self.items) - 1, self.selected + 1)
        elif matches_key(data, ENTER):
            if self.on_select:
                self.on_select(self.selected, self.items[self.selected])
        elif matches_key(data, ESC):
            if self.on_cancel:
                self.on_cancel()

    def render(self, width: int) -> list[str]:
        lines = []
        for i, item in enumerate(self.items):
            prefix = "> " if i == self.selected else "  "
            lines.append(truncate_to_width(prefix + item, width))
        return lines
```

### Handling ANSI Codes

The utility functions correctly handle ANSI escape codes:

```python
from rich.text import Text
from sentarc_tui import visible_width, truncate_to_width

# Rich-styled text
styled = str(Text("Hello", style="bold red"))
width = visible_width(styled)  # 5 (not counting ANSI codes)
truncated = truncate_to_width(styled, 3)  # Preserves ANSI, truncates visible text
```

---

## License

MIT

## See Also

- [sentarc-ai](https://pypi.org/project/sentarc-ai/): Core LLM toolkit
- [sentarc-agent](https://pypi.org/project/sentarc-agent/): Agent framework
- [sentarc-coding-agent](https://pypi.org/project/sentarc-coding-agent/): Interactive coding agent CLI
