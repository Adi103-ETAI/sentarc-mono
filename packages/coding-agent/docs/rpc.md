# RPC Mode

`sentarc_coding_agent.modes.rpc.rpc_mode` exposes arc as a newline-delimited JSON protocol over stdin/stdout so other programs can steer the agent without spawning a TUI.

## Starting the loop

Run `arc --mode rpc` (or add `{"mode": "rpc"}` when embedding) to execute `run_rpc_mode(agent_session)`. The helper subscribes to agent events, forwards them to stdout, and uses `JsonlReader` to stream commands from stdin.

## Transport rules

- **Input**: each line on stdin must be a complete JSON object with at least a `type` field and optional `id` for correlation.
- **Output**: responses and events are emitted as JSON lines via `jsonl.write_jsonl()`.
- The process exits when stdin closes or the hosting application terminates.

## Commands

`rpc_mode.py` currently accepts the following `type` values:

| Command | Payload fields | Behaviour |
| --- | --- | --- |
| `prompt` | `message` | Schedules `agent.prompt(message)` asynchronously and immediately responds with success. |
| `steer` | `message` | Calls `agent.steer()` when available (falls back to `prompt`). |
| `abort` | — | Invokes `agent.abort()` if the agent is streaming. |
| `get_state` | — | Returns `{ thinking_level, is_streaming, message_count, session_id, model }`. |
| `get_messages` | — | Dumps every message currently in `agent._state.messages` as JSON-compatible dicts. |
| `set_model` | `provider`, `model_id`/`modelId` | Resolves the requested model via `sentarc_ai.models.get_model()` and swaps it in-place. |
| `set_thinking_level` | `level` | Updates the agent's thinking level immediately. |
| `get_last_assistant_text` | — | Scans the message list for the most recent assistant text block and returns it. |
| `new_session` | — | Placeholder that simply acknowledges the request (session creation is managed outside RPC for now). |
| `get_available_models` | — | Returns `sentarc_ai.models.list_models()` when available. |

Every reply is shaped by `rpc_types.make_response(command, success, data=None, error=None, id=cmd_id)` and looks like:

```json
{"type":"response","command":"prompt","success":true,"id":"abc123"}
```

Failures set `success:false` and include an `error` string.

## Events

All agent events broadcast by `sentarc_agent.Agent.subscribe()` are forwarded as:

```json
{"type":"event","event":"agent_start"}
{"type":"event","type":"message_start","message":{"role":"assistant"}}
{"type":"event","type":"tool_execution_start","tool_name":"bash","tool_call_id":"..."}
```

Stream consumers can mirror the interactive UI by listening for:

- `agent_start` / `agent_end` – marks inference lifecycles.
- `message_start` / `message_update` / `message_end` – allows live rendering of assistant deltas.
- `tool_execution_start` / `tool_execution_end` – surfaces built-in tool calls (`read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`).

Use this protocol to embed arc inside editors, CI runners, or other orchestrators that prefer JSON over terminal UIs.
