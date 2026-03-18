# sentarc-coding-agent

Arc is a minimal terminal coding agent. Extend it with Python [Extensions](#extensions), [Skills](#skills), [Prompt Templates](#prompt-templates), and [Themes](#themes).

Arc runs in three modes: interactive, print/JSON for scripting, and RPC for process integration.

See [docs/](docs/) for subsystem guides and [examples/](examples/) for ready-made assets.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Providers & Models](#providers--models)
- [Interactive Mode](#interactive-mode)
  - [Commands](#commands)
  - [Keyboard Shortcuts](#keyboard-shortcuts)
- [Sessions](#sessions)
  - [Branching](#branching)
  - [Compaction](#compaction)
- [Settings](#settings)
- [Context Files](#context-files)
- [Customization](#customization)
  - [Prompt Templates](#prompt-templates)
  - [Skills](#skills)
  - [Extensions](#extensions)
  - [Themes](#themes)
- [Programmatic Usage](#programmatic-usage)
- [CLI Reference](#cli-reference)

---

## Installation

Install from PyPI:

```bash
pip install sentarc-coding-agent
arc --help
```

Or for development:

```bash
pip install -e packages/coding-agent
```

---

## Quick Start

```bash
# Authenticate with an API key
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
# or
export GEMINI_API_KEY=...

# Start arc
arc
```

Then just talk to arc. By default, arc gives the model four tools: `read`, `write`, `edit`, and `bash`. The model uses these to fulfill your requests. Add capabilities via [skills](#skills), [prompt templates](#prompt-templates), or [extensions](#extensions).

```bash
# Interactive with initial prompt
arc "List all .py files in src/"

# Non-interactive mode
arc -p "Summarize this codebase"

# Different model
arc --provider openai --model gpt-4o "Help me refactor"

# Include files in prompt
arc @code.py @test.py "Review these files"

# Continue previous session
arc -c

# High thinking level
arc --thinking high "Solve this complex problem"
```

---

## Providers & Models

Arc supports multiple LLM providers. Authenticate via API key environment variables, then select any model from that provider via `/model` command.

**Supported Providers:**

| Provider | Environment Variable |
|----------|---------------------|
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |
| Google Vertex AI | `GOOGLE_CLOUD_PROJECT` + `GOOGLE_CLOUD_LOCATION` |
| Amazon Bedrock | `AWS_REGION` (uses standard boto3 auth) |
| OpenAI-compatible | `OPENAI_API_KEY` + custom `base_url` |

**List available models:**

```bash
arc --list-models
arc --list-models claude  # Filter by search term
```

**Custom models:** Add providers via `~/.arc/agent/models.json` if they speak a supported API (OpenAI, Anthropic, Google).

---

## Interactive Mode

The interface from top to bottom:

- **Messages** - Your messages, assistant responses, tool calls and results
- **Editor** - Where you type your prompts
- **Footer** - Current model, thinking level, session info

### Commands

Type `/` in the editor to trigger commands. [Extensions](#extensions) can register custom commands, and [skills](#skills) are available as `/skill:name`.

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/clear` | Clear conversation history |
| `/model <spec>` | Switch models (e.g., `/model openai/gpt-4o`) |
| `/thinking <level>` | Set thinking level (off, minimal, low, medium, high, xhigh) |
| `/compact [prompt]` | Manually compact context, optional custom instructions |
| `/branch <id>` | Branch from a specific entry ID |
| `/export [file]` | Export session to HTML file |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Ctrl+J | Submit message |
| Ctrl+C | Abort current operation / Clear editor |
| Ctrl+C twice | Quit |
| F1 | Show help |
| Ctrl+L | Clear screen |

---

## Sessions

Sessions are stored as JSONL files with a tree structure. Each entry has an `id` and `parentId`, enabling in-place branching without creating new files.

### Management

Sessions auto-save to `~/.arc/agent/sessions/` organized by working directory.

```bash
arc -c                  # Continue most recent session
arc -r                  # Browse and select from past sessions
arc --no-session        # Ephemeral mode (don't save)
arc --session <path>    # Use specific session file
```

### Branching

Use `/branch <entry_id>` to create alternative conversation paths. All history is preserved in a single file.

### Compaction

Long sessions can exhaust context windows. Compaction summarizes older messages while keeping recent ones.

**Manual:** `/compact` or `/compact <custom instructions>`

**What gets compacted:**
- Goal and progress summary
- Key files read/edited
- Important context
- Next steps

The full history remains in the JSONL file; use branching to revisit earlier states.

---

## Settings

Edit settings directly in JSON:

| Location | Scope |
|----------|-------|
| `~/.arc/agent/settings.json` | Global (all projects) |
| `.arc/settings.json` | Project (overrides global) |

**Available settings:**

```json
{
  "provider": "google",
  "model": "gemini-2.5-flash",
  "thinking": "off",
  "quiet_startup": false,
  "tools": ["read", "bash", "edit", "write"]
}
```

---

## Context Files

Arc loads `AGENTS.md` at startup from:
- `~/.arc/agent/AGENTS.md` (global)
- Parent directories (walking up from cwd)
- Current directory

Use for project instructions, conventions, common commands. All matching files are concatenated into the system prompt.

### System Prompt

Replace the default system prompt with `.arc/SYSTEM.md` (project) or `~/.arc/agent/SYSTEM.md` (global). Append without replacing via `APPEND_SYSTEM.md`.

---

## Customization

Detailed subsystem guides live under [docs/](docs/) (extensions, skills, sessions, compaction, RPC, etc.), and reusable assets will land in [examples/](examples/) as they are published.

### Prompt Templates

Reusable prompts as Markdown files. Type `/templatename` to expand.

```markdown
<!-- ~/.arc/agent/prompts/review.md -->
---
name: review
description: Code review template
---
Review this code for bugs, security issues, and performance problems.
Focus on: {{focus}}
```

Place in `~/.arc/agent/prompts/` or `.arc/prompts/`.

### Skills

On-demand capability packages. Invoke via `/skill:name` or let the agent load them automatically.

```markdown
<!-- ~/.arc/agent/skills/my-skill/SKILL.md -->
---
name: my-skill
description: Use this skill when the user asks about X.
---
# My Skill

## Steps
1. Do this
2. Then that
```

Place in `~/.arc/agent/skills/` or `.arc/skills/`.

### Extensions

Python modules that extend arc with custom tools, commands, keyboard shortcuts, and event handlers.

```python
# ~/.arc/agent/extensions/my_extension.py

EXTENSION_NAME = "my-extension"

COMMANDS = [
    {
        "name": "stats",
        "description": "Show session statistics",
        "execute": lambda ctx: print_stats(ctx)
    }
]

def on_start(ctx):
    """Called when arc starts."""
    pass

def on_tool_call(ctx, tool_name, args):
    """Called before each tool execution."""
    pass

def on_agent_end(ctx, messages):
    """Called when agent completes."""
    pass
```

**What's possible:**
- Custom tools (or replace built-in tools)
- Custom slash commands
- Permission gates and path protection
- Git checkpointing and auto-commit
- Custom UI components
- ...anything you can code in Python

Place in `~/.arc/agent/extensions/` or `.arc/extensions/`.

### Themes

Built-in: `dark`, `light`. Custom themes can be placed in `~/.arc/agent/themes/` or `.arc/themes/`.

---

## Programmatic Usage

Python consumers can call `sentarc_agent.Agent` directly; see `examples/sdk/` for runnable scripts that cover a minimal prompt, tool-enabled runs, and session resume.

### RPC Mode

For non-Python integrations, use RPC mode over stdin/stdout:

```bash
arc --mode rpc
```

**RPC Commands:**

| Command | Description |
|---------|-------------|
| `prompt` | Send a prompt |
| `steer` | Interrupt with a steering message |
| `abort` | Abort current operation |
| `get_state` | Get agent state |
| `get_messages` | Get conversation messages |
| `set_model` | Switch model |
| `set_thinking_level` | Change thinking level |
| `new_session` | Start new session |
| `get_available_models` | List models |

### Print/JSON Mode

For scripting:

```bash
# Text output
arc -p "Summarize this file" @file.py

# JSON output (all events as JSONL)
arc --mode json "What files are here?"
```

---

## CLI Reference

```bash
arc [options] [@files...] [messages...]
```

### Model Options

| Option | Description |
|--------|-------------|
| `--provider <name>` | Provider (anthropic, openai, google, etc.) |
| `--model <pattern>` | Model pattern or ID |
| `--api-key <key>` | API key (overrides env vars) |
| `--thinking <level>` | `off`, `minimal`, `low`, `medium`, `high`, `xhigh` |
| `--models <patterns>` | Comma-separated patterns for model cycling |
| `--list-models [search]` | List available models |

### Session Options

| Option | Description |
|--------|-------------|
| `-c`, `--continue` | Continue most recent session |
| `-r`, `--resume` | Browse and select session |
| `--session <path>` | Use specific session file |
| `--session-dir <dir>` | Custom session storage directory |
| `--no-session` | Ephemeral mode (don't save) |

### Tool Options

| Option | Description |
|--------|-------------|
| `--tools <list>` | Enable specific tools (default: `read,bash,edit,write`) |
| `--no-tools` | Disable all built-in tools |

**Available tools:** `read`, `bash`, `edit`, `write`, `grep`, `find`, `ls`

| Tool | Description |
|------|-------------|
| `read` | Read file contents (text and images) |
| `bash` | Execute bash commands with streaming output |
| `edit` | Surgical find-and-replace edits |
| `write` | Create or overwrite files |
| `grep` | Search file contents (uses ripgrep if available) |
| `find` | Find files by glob pattern (uses fd if available) |
| `ls` | List directory contents |

### Mode Options

| Option | Description |
|--------|-------------|
| (default) | Interactive mode |
| `-p`, `--print` | Print response and exit |
| `--mode json` | Output all events as JSON lines |
| `--mode rpc` | RPC mode for process integration |
| `--export <file>` | Export session to HTML |

### Resource Options

| Option | Description |
|--------|-------------|
| `-e`, `--extension <path>` | Load extension (repeatable) |
| `--no-extensions` | Disable extension discovery |
| `--skill <path>` | Load skill (repeatable) |
| `--no-skills` | Disable skill discovery |
| `--prompt-template <path>` | Load prompt template (repeatable) |
| `--no-prompt-templates` | Disable prompt template discovery |
| `--theme <path>` | Load theme |
| `--no-themes` | Disable theme discovery |

### Other Options

| Option | Description |
|--------|-------------|
| `--system-prompt <text>` | Replace default prompt |
| `--append-system-prompt <text>` | Append to system prompt |
| `--verbose` | Force verbose startup |
| `-h`, `--help` | Show help |
| `-v`, `--version` | Show version |

### File Arguments

Prefix files with `@` to include in the message:

```bash
arc @prompt.md "Answer this"
arc -p @screenshot.png "What's in this image?"
arc @code.py @test.py "Review these files"
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ARC_CODING_AGENT_DIR` | Override config directory (default: `~/.arc/agent`) |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `GEMINI_API_KEY` | Google Gemini API key |

---

## License

MIT

## See Also

- [sentarc-ai](https://pypi.org/project/sentarc-ai/): Core LLM toolkit
- [sentarc-agent](https://pypi.org/project/sentarc-agent/): Agent framework
- [sentarc-tui](https://pypi.org/project/sentarc-tui/): Terminal UI components
