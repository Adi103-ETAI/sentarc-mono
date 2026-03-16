# sentarc-coding-agent

Sentarc coding agent CLI — the `arc` command.

## Usage

```bash
arc [options] [@files...] [messages...]
```

## Options

- `--provider <name>` — Provider name (default: google)
- `--model <pattern>` — Model pattern or ID
- `--api-key <key>` — API key
- `--system-prompt <text>` — Custom system prompt
- `--append-system-prompt <text>` — Append to system prompt
- `--mode <mode>` — Output mode: text (default), json, rpc
- `--print, -p` — Non-interactive mode
- `--continue, -c` — Continue previous session
- `--resume, -r` — Select a session to resume
- `--session <path>` — Use specific session file
- `--no-session` — Don't save session
- `--tools <tools>` — Comma-separated tool names (read,bash,edit,write,grep,find,ls)
- `--no-tools` — Disable all tools
- `--thinking <level>` — Thinking level: off, minimal, low, medium, high, xhigh
- `--list-models [search]` — List available models
- `--help, -h` — Show help
- `--version, -v` — Show version

## Available Tools (default: read, bash, edit, write)

- `read` — Read file contents
- `bash` — Execute bash commands
- `edit` — Edit files with find/replace
- `write` — Write files (creates/overwrites)
- `grep` — Search file contents (read-only)
- `find` — Find files by glob pattern (read-only)
- `ls` — List directory contents (read-only)
