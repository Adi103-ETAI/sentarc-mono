# Sessions

`sentarc_coding_agent.core.session_manager.SessionManager` persists every conversation as a JSONL tree so you can resume work, branch, and summarize long histories without losing context.

## Storage layout

- Base directory: `~/.arc/agent/sessions/` (respect `ARC_CODING_AGENT_DIR`).
- Subdirectory per project: the current working directory encoded with `/` → `--` (`_encode_cwd`).
- Files: `<timestamp>_<session-id>.jsonl`, created only after the first assistant response so empty sessions are not written.

Factory helpers:

| Method | Purpose |
| --- | --- |
| `SessionManager.create(cwd, session_dir=None)` | Start a new on-disk session. |
| `SessionManager.continue_recent(cwd, session_dir=None)` | Reopen the most recently touched file for the project. |
| `SessionManager.open(path, session_dir=None)` | Load a specific JSONL file. |
| `SessionManager.in_memory(cwd=None)` | Use the APIs without touching disk (tests, RPC sandboxes). |

## Entry types

Every non-header line includes an `id`, `parentId`, `timestamp`, and `type`:

| Type | Added by | Why it matters |
| --- | --- | --- |
| `session` | Constructor | File header storing version, cwd, and master id. |
| `message` | `append_message()` | User, assistant, and tool result payloads. |
| `custom_message` | `append_custom_message()` | Rich text emitted by extensions or integrations. |
| `thinking_level_change` | `/thinking`, RPC `set_thinking_level` | Records long-lived reasoning preferences. |
| `model_change` | `/model`, RPC `set_model` | Tracks provider/model switches. |
| `compaction` | `append_compaction()` | Describes a compaction summary (`summary`, `tokensBefore`, `firstKeptEntryId`). |
| `branch_summary` | `append_branch_summary()` | Documents when you fork from `fromId`. |
| `label` | `set_label()` | Stores arbitrary tags for history viewers. |
| `custom` | `append_custom()` | Free-form metadata for future tooling. |

`build_session_context()` walks parent pointers from the current `leaf_id`, converts entries back into messages (including compaction or branch summary stubs), and returns `{"messages": [...], "thinkingLevel": ..., "model": ...}` for the agent startup.

## Branching & slash commands

`SessionManager.branch(entry_id)` rewinds the leaf pointer so subsequent `append_message()` calls fork from that earlier moment. `branch_with_summary()` both rewinds and appends a `branch_summary` describing the reason.

`core.slash_commands.handle_slash_command()` wires built-in commands that manipulate the session in every UI:

| Command | Effect |
| --- | --- |
| `/help` | Lists all built-in commands. |
| `/clear` | Calls `reset_leaf()` and clears the agent's in-memory history. |
| `/compact [notes]` | Placeholder until compaction is fully plumbed through the Python UI. |
| `/model <provider/model>` | Uses `SessionManager.append_model_change()` to persist the switch. |
| `/thinking <level>` | Persists the new thinking level. |
| `/branch <entry-id>` | Calls `branch()`; errors if the id is unknown. |
| `/export [file]` | Placeholder for future HTML exports. |

Extensions can add more slash commands via the extension API, and `on_tool_call` hooks can also append custom entries for auditing built-in tools (`read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`).

## Listing & browsing sessions

- `SessionManager.list(cwd, session_dir=None)` gathers metadata for the current project (message count, timestamps, first user prompt). The `arc -r/--resume` TUI uses this to show a picker.
- `SessionManager.list_all()` scans every encoded directory under `~/.arc/agent/sessions` and sorts by `modified`.

Use these helpers to build dashboards, implement custom pruning policies, or feed summaries into other systems.
