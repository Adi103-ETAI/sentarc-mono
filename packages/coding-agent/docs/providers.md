# Providers & Models

Arc defers provider-specific auth and transport to `sentarc_ai`, while `sentarc_coding_agent.core.model_resolver` handles parsing user input (`--model`, `/model`, RPC) into the `(provider, model_id, thinking_level)` triple the agent needs.

## Environment variables

Set the appropriate secrets before launching arc:

| Provider | Environment variables |
| --- | --- |
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI or OpenAI-compatible | `OPENAI_API_KEY` (configure custom base URLs inside your `sentarc_ai` setup if needed) |
| Google Gemini | `GEMINI_API_KEY` |
| Google Vertex AI | `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` (plus standard Google ADC credentials) |
| Amazon Bedrock | `AWS_REGION` and standard AWS credentials |

`--api-key` overrides the detected key for a single invocation. OAuth-style providers can store tokens in `~/.arc/agent/auth.json` via `core.auth_storage`.

## Selecting a model

1. CLI parsing collects `--provider`, `--model`, `--thinking`, and optional `--models` (future cycling list).
2. `model_resolver.parse_model_spec()` accepts inputs such as `claude-3.5-sonnet`, `openai/gpt-4o`, or `anthropic/claude-3.5-sonnet:high` (suffix sets the thinking level).
3. The resolved tuple feeds into `sentarc_ai.models.get_model()`; the resulting object is stored on the agent state and referenced by the UI footer.
4. If the chosen model encodes a preferred thinking level, it wins unless you pass `--thinking` explicitly.

Global defaults live in `settings.json` (`provider`, `model`, `thinking`).

## Runtime switches

- Slash command `/model <provider/model[:thinking]>` resolves and persists the change via `SessionManager.append_model_change()`.
- Slash command `/thinking <level>` records the new thinking level through `append_thinking_level_change()`.
- RPC commands `set_model` and `set_thinking_level` expose the same functionality to external processes.

Because these actions are logged in the session file, you get a full audit trail when resuming work.

## Listing & registering models

- `arc --list-models [filter]` prints everything returned by `core.model_registry.get_all_models()`.
- `~/.arc/agent/models.json` can define additional entries (list of `{ "provider": "openai", "id": "local-gpt4o", "name": "My self-hosted endpoint" }`).
- Any metadata returned by `sentarc_ai.models.list_models()` is passed straight through.

Use the CLI and RPC commands together with these files to keep provider usage consistent across interactive, print, and automated workflows.
