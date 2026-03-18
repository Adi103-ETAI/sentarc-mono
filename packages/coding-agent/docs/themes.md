# Themes

Interactive mode styles are driven by `sentarc_coding_agent.modes.interactive.theme.InteractiveTheme`, a dataclass that feeds colors and emphasis values into Textual widgets (`UserMessageWidget`, `AssistantMessageWidget`, `ToolExecutionWidget`, and the footer).

## Built-in themes

Two presets ship with arc:

- `dark` (`DARK_THEME`) – blue user text, green assistant text, yellow tool progress, dark footer background.
- `light` (`LIGHT_THEME`) – brighter assistants, light footer background, italic thinking indicator.

`load_theme("dark")` or `load_theme("light")` returns these predefined structures.

## Custom themes

`load_theme(name, themes_dir)` follows this process:

1. Return `DARK_THEME`/`LIGHT_THEME` for the built-in names.
2. Otherwise, look for `<themes_dir>/<name>.json` and load any keys that match the dataclass fields.

`get_custom_themes_dir()` points to `~/.arc/agent/themes/` (or `<ARC_CODING_AGENT_DIR>/themes`). Drop JSON files there with any subset of the available properties:

| Field | Default |
| --- | --- |
| `user_message` | `"blue"` |
| `assistant_message` | `"green"` |
| `tool_running` / `tool_success` / `tool_error` | `"yellow"` / `"green"` / `"red"` |
| `thinking` | `"dim cyan"` |
| `input_border` / `input_focused_border` | `"blue"` / `"bright_blue"` |
| `footer_bg` / `footer_text` | `"#2d2d2d"` / `"white"` |
| `accent`, `muted`, `error`, `warning`, `success` | Utility colors used across widgets |

Example override:

```json
{
  "assistant_message": "bright_cyan",
  "footer_bg": "#111",
  "thinking": "italic green"
}
```

## Using themes today

`cli.args.parse_args()` already understands `--theme <path>` and `--no-themes`, but the current Python CLI always loads `load_theme("dark", Path(get_custom_themes_dir()))` when launching `run_interactive_mode()`. Embedding applications can call `load_theme()` directly (or supply the parsed theme) before handing control to the TUI.

When extensions detect `ctx.has_ui`, they can also pull styling cues from the current `InteractiveTheme` to render consistent notifications via `ctx.notify()`.
