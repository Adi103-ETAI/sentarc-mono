# Extensions

Arc discovers Python modules and wires their hooks into the runtime so you can intercept startup, tools, and session activity. Discovery happens in `sentarc_coding_agent.core.extensions.loader`; hook execution lives in `core.extensions.runner`.

## Discovery & opt-out

1. **Global** – every `~/.arc/agent/tools/*.py` file is loaded (set `ARC_CODING_AGENT_DIR` to relocate the root).
2. **Project** – hosts may pass a `project_dir` to `discover_extensions()`, which scans `<project>/.arc/extensions/`. The current CLI does not supply this argument yet, but embedding apps can.
3. **Explicit** – each `-e/--extension <path>` argument (relative paths resolve against the current working directory).

Use `--no-extensions`/`-ne` to skip the entire system.

## Module contract

`core.extensions.types.Extension` expects these exports:

| Export | Type | Notes |
| --- | --- | --- |
| `EXTENSION_NAME` | `str` | Human-friendly name; defaults to the filename stem. |
| `EXTENSION_FLAGS` | `dict[str, {"type": "boolean"\|"string", "description": str}]` | Optional CLI switches. Parsers that pass `extension_flags` to `cli.args.parse_args()` will accept them alongside core flags. |
| `COMMANDS` | `list[ExtensionCommand]` | Slash commands rendered in `/help`; handlers may be sync or `async` coroutines. |
| `on_start(ctx)` | callable | Runs right after the module is imported. |
| `on_before_agent_start(ctx)` | callable | Last chance to mutate CLI args or environment variables before the agent spins up. |
| `on_agent_start(ctx)` / `on_agent_end(ctx, messages)` | callable | Observe each agent run. `messages` mirrors the payload broadcast by `sentarc_agent`. |
| `on_session_start(ctx)` / `on_session_switch(ctx)` / `on_session_end(ctx)` | callable | Track lifecycle events emitted by the session manager. |
| `on_message(ctx, message)` | callable | Fires for every message routed through `SessionManager.append_message`. |
| `on_tool_call(ctx, tool_name, args)` | callable | Called before each built-in tool (`read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`) executes. Raise if you need to abort. |

`run_hook()` awaits coroutine hooks, ignores return values, and prints warnings instead of crashing when a hook raises.

## Extension context

Every hook receives an `ExtensionContext`.

| Field | Type | Description |
| --- | --- | --- |
| `args` | `dict[str, Any]` | Parsed CLI arguments plus extension-defined flag values. |
| `cwd` | `str` | Working directory that tools resolve against. |
| `session_manager` | `SessionManager \| None` | Read/append session entries, branch, or compact history. |
| `agent` | `sentarc_agent.agent.Agent` | Access the live agent for subscriptions or manual prompts. |
| `has_ui` | `bool` | Indicates whether the Textual UI is active. |
| `notify` | `Callable[[str, str], None] \| None` | Present toast notifications in interactive mode. Always guard for `None` in print/RPC modes. |

## Commands & flags

Populate `COMMANDS` with dicts that look like `ExtensionCommand` and they become `/your-command` entries in the input box:

```python
async def stats(args: str, ctx):
    branch = ctx.session_manager.get_branch() if ctx.session_manager else []
    return f"{len(branch)} entries in current branch"

COMMANDS = [{
    "name": "stats",
    "description": "Show branch length",
    "handler": stats,
}]
```

Flags declared in `EXTENSION_FLAGS` land inside `ctx.args` (boolean flags become `True`, string flags capture the next token). Host applications must pass the flag schema to `parse_args()` for this to work.

## Tool interception & custom tools

Use `on_tool_call()` to enforce policies before a tool runs:

```python
def on_tool_call(ctx, tool_name, args):
    if tool_name == "bash" and "rm -rf" in args.get("command", ""):
        raise RuntimeError("Blocked destructive command")
```

Extensions can also attach arbitrary `custom_tools` attributes to expose new tools to the agent—`cli.__init__` checks for this attribute and appends the returned tool definitions to the built-in set.

## Example

```python
# ~/.arc/agent/tools/audit.py
EXTENSION_NAME = "audit"

async def on_agent_end(ctx, messages=None):
    if ctx.session_manager and messages:
        ctx.session_manager.append_custom_message(
            "audit", {"count": len(messages)}, display=False,
        )

COMMANDS = [{
    "name": "audit",
    "description": "Print the last leaf entry",
    "handler": lambda args, ctx: ctx.session_manager.get_leaf_entry() if ctx.session_manager else "no session",
}]
```

Drop the file into a discovery directory, restart arc, and the command appears alongside the built-in `/help`, `/branch`, `/model`, and `/thinking` slash commands.
