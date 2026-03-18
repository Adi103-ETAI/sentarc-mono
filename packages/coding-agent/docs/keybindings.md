# Keybindings

`ArcInteractiveApp` (defined in `modes/interactive/interactive_mode.py`) is a Textual application that wires widgets and shortcuts so you can steer the agent entirely from the keyboard.

- `VerticalScroll#messages` hosts `UserMessageWidget`, `AssistantMessageWidget`, and `ToolExecutionWidget` instances.
- `TextArea#input-editor` accepts prompts and slash commands (e.g., `/help`, `/branch`, `/model`).
- `FooterWidget` shows the active provider/model plus thinking level and streaming state.

## Global bindings

`ArcInteractiveApp.BINDINGS` registers the shortcuts Textual displays in the footer:

| Keys | Action | Behaviour |
| --- | --- | --- |
| `Ctrl+C` | `abort_or_quit` | Aborts the current run if streaming; otherwise exits the UI. |
| `F1` | `show_help` | Pops a notification describing every binding. |
| `Ctrl+L` | `clear_screen` | Removes all message widgets from the scrollback. |

Because the bindings include `show=True`, Textual renders the labels (“Abort/Quit”) automatically.

## Editor shortcuts

`on_key()` also intercepts `Ctrl+J` to submit the buffer even though the editor allows multi-line input:

```python
if event.key == "ctrl+j":
    self._send_message()
    event.prevent_default()
```

`_send_message()` mounts a `UserMessageWidget`, scrolls to the bottom, clears the editor, and schedules `agent.prompt(text)` asynchronously. Slash commands share the same pipeline—type `/model anthropic/claude-3.5-sonnet` or `/thinking high` and press `Ctrl+J`.

## Notifications & extensions

The actions `action_show_help()` and `action_clear_screen()` rely on Textual notifications. When extensions detect `ctx.has_ui`, they can call `ctx.notify("message", level="info")` to display guidance that complements the built-in slash commands and keybindings.

Use these shortcuts together with the default tools (`read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`) to keep your hands on the keyboard while iterating.
