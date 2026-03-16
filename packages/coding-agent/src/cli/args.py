"""
CLI argument parsing and help display.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from sentarc_coding_agent.config import APP_NAME, CONFIG_DIR_NAME, ENV_AGENT_DIR

VALID_THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh")
VALID_MODES = ("text", "json", "rpc")
VALID_TOOLS = ("read", "bash", "edit", "write", "grep", "find", "ls")


def is_valid_thinking_level(level: str) -> bool:
    return level in VALID_THINKING_LEVELS


def parse_args(
    argv: List[str],
    extension_flags: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Parse CLI arguments, mirroring the TypeScript parseArgs() function.
    Returns a dict with the same keys as the TS Args interface (snake_case).
    """
    result: Dict[str, Any] = {
        "messages": [],
        "file_args": [],
        "unknown_flags": {},
    }

    i = 0
    while i < len(argv):
        arg = argv[i]

        if arg in ("--help", "-h"):
            result["help"] = True
        elif arg in ("--version", "-v"):
            result["version"] = True
        elif arg == "--mode" and i + 1 < len(argv):
            i += 1
            mode = argv[i]
            if mode in VALID_MODES:
                result["mode"] = mode
        elif arg in ("--continue", "-c"):
            result["continue"] = True
        elif arg in ("--resume", "-r"):
            result["resume"] = True
        elif arg == "--provider" and i + 1 < len(argv):
            i += 1
            result["provider"] = argv[i]
        elif arg == "--model" and i + 1 < len(argv):
            i += 1
            result["model"] = argv[i]
        elif arg == "--api-key" and i + 1 < len(argv):
            i += 1
            result["api_key"] = argv[i]
        elif arg == "--system-prompt" and i + 1 < len(argv):
            i += 1
            result["system_prompt"] = argv[i]
        elif arg == "--append-system-prompt" and i + 1 < len(argv):
            i += 1
            result["append_system_prompt"] = argv[i]
        elif arg == "--no-session":
            result["no_session"] = True
        elif arg == "--session" and i + 1 < len(argv):
            i += 1
            result["session"] = argv[i]
        elif arg == "--session-dir" and i + 1 < len(argv):
            i += 1
            result["session_dir"] = argv[i]
        elif arg == "--models" and i + 1 < len(argv):
            i += 1
            result["models"] = [s.strip() for s in argv[i].split(",")]
        elif arg == "--no-tools":
            result["no_tools"] = True
        elif arg == "--tools" and i + 1 < len(argv):
            i += 1
            tool_names = [s.strip() for s in argv[i].split(",")]
            valid: List[str] = []
            for name in tool_names:
                if name in VALID_TOOLS:
                    valid.append(name)
                else:
                    print(
                        f"Warning: Unknown tool \"{name}\". Valid tools: {', '.join(VALID_TOOLS)}",
                        file=sys.stderr,
                    )
            result["tools"] = valid
        elif arg == "--thinking" and i + 1 < len(argv):
            i += 1
            level = argv[i]
            if is_valid_thinking_level(level):
                result["thinking"] = level
            else:
                print(
                    f"Warning: Invalid thinking level \"{level}\". Valid values: {', '.join(VALID_THINKING_LEVELS)}",
                    file=sys.stderr,
                )
        elif arg in ("--print", "-p"):
            result["print"] = True
        elif arg == "--export" and i + 1 < len(argv):
            i += 1
            result["export"] = argv[i]
        elif arg in ("--extension", "-e") and i + 1 < len(argv):
            i += 1
            result.setdefault("extensions", []).append(argv[i])
        elif arg in ("--no-extensions", "-ne"):
            result["no_extensions"] = True
        elif arg == "--skill" and i + 1 < len(argv):
            i += 1
            result.setdefault("skills", []).append(argv[i])
        elif arg == "--prompt-template" and i + 1 < len(argv):
            i += 1
            result.setdefault("prompt_templates", []).append(argv[i])
        elif arg == "--theme" and i + 1 < len(argv):
            i += 1
            result.setdefault("themes", []).append(argv[i])
        elif arg in ("--no-skills", "-ns"):
            result["no_skills"] = True
        elif arg in ("--no-prompt-templates", "-np"):
            result["no_prompt_templates"] = True
        elif arg == "--no-themes":
            result["no_themes"] = True
        elif arg == "--list-models":
            # Check if next arg is a search pattern (not a flag or file arg)
            if i + 1 < len(argv) and not argv[i + 1].startswith("-") and not argv[i + 1].startswith("@"):
                i += 1
                result["list_models"] = argv[i]
            else:
                result["list_models"] = True
        elif arg == "--verbose":
            result["verbose"] = True
        elif arg == "--offline":
            result["offline"] = True
        elif arg.startswith("@"):
            result["file_args"].append(arg[1:])  # Remove @ prefix
        elif arg.startswith("--") and extension_flags:
            flag_name = arg[2:]
            flag_type = extension_flags.get(flag_name)
            if flag_type == "boolean":
                result["unknown_flags"][flag_name] = True
            elif flag_type == "string" and i + 1 < len(argv):
                i += 1
                result["unknown_flags"][flag_name] = argv[i]
        elif not arg.startswith("-"):
            result["messages"].append(arg)

        i += 1

    return result


def print_help() -> None:
    """Print help text."""
    print(f"""{APP_NAME} - AI coding assistant with read, bash, edit, write tools

Usage:
  {APP_NAME} [options] [@files...] [messages...]

Options:
  --provider <name>              Provider name (default: google)
  --model <pattern>              Model pattern or ID
  --api-key <key>                API key
  --system-prompt <text>         System prompt
  --append-system-prompt <text>  Append to system prompt
  --mode <mode>                  Output mode: text (default), json, rpc
  --print, -p                    Non-interactive mode
  --continue, -c                 Continue previous session
  --resume, -r                   Select a session to resume
  --session <path>               Use specific session file
  --session-dir <dir>            Directory for session storage
  --no-session                   Don't save session
  --models <patterns>            Comma-separated model patterns for cycling
  --no-tools                     Disable all built-in tools
  --tools <tools>                Comma-separated list of tools to enable
                                 Available: read, bash, edit, write, grep, find, ls
  --thinking <level>             Set thinking level: off, minimal, low, medium, high, xhigh
  --extension, -e <path>         Load an extension file
  --no-extensions, -ne           Disable extension discovery
  --skill <path>                 Load a skill file or directory
  --no-skills, -ns               Disable skills discovery
  --prompt-template <path>       Load a prompt template
  --no-prompt-templates, -np     Disable prompt template discovery
  --theme <path>                 Load a theme
  --no-themes                    Disable theme discovery
  --export <file>                Export session file to HTML and exit
  --list-models [search]         List available models
  --verbose                      Force verbose startup
  --offline                      Disable startup network operations
  --help, -h                     Show this help
  --version, -v                  Show version number

Examples:
  # Interactive mode
  {APP_NAME}

  # Non-interactive mode
  {APP_NAME} -p "List all .py files in src/"

  # Include files in message
  {APP_NAME} @prompt.md "What does this do?"

  # Continue previous session
  {APP_NAME} --continue "What did we discuss?"

  # Use different model
  {APP_NAME} --provider openai --model gpt-4o "Help me refactor"

Environment Variables:
  ANTHROPIC_API_KEY                - Anthropic Claude API key
  OPENAI_API_KEY                   - OpenAI GPT API key
  GEMINI_API_KEY                   - Google Gemini API key
  {ENV_AGENT_DIR:<32} - Session storage directory (default: ~/{CONFIG_DIR_NAME}/agent)
""")
