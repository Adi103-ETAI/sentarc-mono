# Settings

`sentarc_coding_agent.core.settings_manager` centralizes arc's defaults in a single JSON file so the CLI, RPC server, and fallback REPL can agree on provider, model, tools, and thinking level.

## Location & overrides

`get_settings_path()` resolves to `~/.arc/agent/settings.json`. Set `ARC_CODING_AGENT_DIR` to relocate the entire config tree, which changes the settings path to `<dir>/settings.json`. Project-level overrides are not implemented in the Python port yet, so every run shares the global file.

## Dataclass schema

`Settings` is a dataclass whose values double as defaults when the file is missing or unreadable:

| JSON key | Default | Purpose |
| --- | --- | --- |
| `provider` | `"google"` | Used when `--provider` is absent. Passed to `model_resolver.resolve_model()`. |
| `model` | `"gemini-2.5-flash"` | Base model pattern. CLI `--model` or slash command `/model` can override per run. |
| `thinking` | `"off"` | Baseline thinking level until CLI flags, session history, or `/thinking` change it. |
| `quietStartup` | `false` | Reserved for silencing verbose banners in future UIs. Currently read and written as-is. |
| `tools` | `["read", "bash", "edit", "write"]` | Default tool list whenever `--tools`/`--no-tools` are not supplied. |

`load_settings()` returns a populated dataclass; `save_settings()` writes the same keys back (camelCase for `quietStartup`). Extra keys are stored in the `extra` dict for forward compatibility.

## How values flow through the app

1. `cli.__init__` calls `load_settings()` before parsing most flags. Unless you pass explicit CLI overrides, the settings file determines the provider, model, thinking level, and enabled tool list.
2. `core.agent_session.run_agent_session()` does the same thing inside the fallback text UI.
3. The system prompt builder receives the final tool list (`read`, `bash`, `edit`, `write`, plus optional `grep`, `find`, `ls`) so every downstream model call gets accurate capabilities.

Edit the JSON file directly, save it, and restart arc to apply new defaults.
