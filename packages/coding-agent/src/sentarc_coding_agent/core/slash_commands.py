"""
Built-in slash commands: /help, /compact, /branch, /export, /clear, /model, /thinking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class SlashCommandContext:
    args: str
    session_manager: Any
    agent: Any
    cwd: str


@dataclass
class SlashCommand:
    name: str
    description: str
    handler: Callable[[SlashCommandContext], Any]


async def _help_command(ctx: SlashCommandContext) -> str:
    lines = [
        "Available slash commands:",
        "  /help            — Show this help",
        "  /clear           — Clear conversation history",
        "  /compact         — Compact conversation history",
        "  /model <spec>    — Switch model (e.g. /model openai/gpt-4o)",
        "  /thinking <lvl>  — Set thinking level (off/minimal/low/medium/high/xhigh)",
        "  /branch <id>     — Branch from a specific entry",
        "  /export [file]   — Export session to HTML",
    ]
    return "\n".join(lines)


async def _clear_command(ctx: SlashCommandContext) -> str:
    if ctx.session_manager:
        ctx.session_manager.reset_leaf()
    if ctx.agent:
        ctx.agent.state.messages.clear()
    return "Conversation cleared."


async def _compact_command(ctx: SlashCommandContext) -> str:
    return "Compaction not yet implemented in this mode."


async def _model_command(ctx: SlashCommandContext) -> str:
    from sentarc_coding_agent.core.model_resolver import resolve_model
    spec = ctx.args.strip()
    if not spec:
        return "Usage: /model <provider>/<model-id>[:<thinking>]"
    provider, model_id, thinking = resolve_model(None, spec)
    try:
        from sentarc_ai.models import get_model
        model_def = get_model(provider, model_id)
        if ctx.agent:
            ctx.agent.state.model = model_def
        if ctx.session_manager:
            ctx.session_manager.append_model_change(provider, model_id)
        return f"Switched to model: {provider}/{model_id}"
    except Exception as e:
        return f"Error switching model: {e}"


async def _thinking_command(ctx: SlashCommandContext) -> str:
    valid = ("off", "minimal", "low", "medium", "high", "xhigh")
    level = ctx.args.strip().lower()
    if level not in valid:
        return f"Invalid thinking level. Valid: {', '.join(valid)}"
    if ctx.agent:
        ctx.agent.state.thinking_level = level
    if ctx.session_manager:
        ctx.session_manager.append_thinking_level_change(level)
    return f"Thinking level set to: {level}"


async def _branch_command(ctx: SlashCommandContext) -> str:
    entry_id = ctx.args.strip()
    if not entry_id:
        return "Usage: /branch <entry-id>"
    if ctx.session_manager:
        try:
            ctx.session_manager.branch(entry_id)
            return f"Branched from entry: {entry_id}"
        except Exception as e:
            return f"Error: {e}"
    return "No session manager available."


async def _export_command(ctx: SlashCommandContext) -> str:
    return "Export not yet implemented."


SLASH_COMMANDS: List[SlashCommand] = [
    SlashCommand("help", "Show help", _help_command),
    SlashCommand("clear", "Clear conversation history", _clear_command),
    SlashCommand("compact", "Compact conversation history", _compact_command),
    SlashCommand("model", "Switch model", _model_command),
    SlashCommand("thinking", "Set thinking level", _thinking_command),
    SlashCommand("branch", "Branch from entry", _branch_command),
    SlashCommand("export", "Export session to HTML", _export_command),
]

_COMMAND_MAP: Dict[str, SlashCommand] = {cmd.name: cmd for cmd in SLASH_COMMANDS}


async def handle_slash_command(
    input_text: str,
    context: SlashCommandContext,
) -> Optional[str]:
    """
    Handle a slash command input.
    Returns response string if handled, None if not a slash command.
    """
    if not input_text.startswith("/"):
        return None

    parts = input_text[1:].split(" ", 1)
    cmd_name = parts[0].lower()
    cmd_args = parts[1] if len(parts) > 1 else ""

    cmd = _COMMAND_MAP.get(cmd_name)
    if not cmd:
        return f"Unknown command: /{cmd_name}. Type /help for available commands."

    ctx = SlashCommandContext(
        args=cmd_args,
        session_manager=context.session_manager,
        agent=context.agent,
        cwd=context.cwd,
    )
    return await cmd.handler(ctx)
