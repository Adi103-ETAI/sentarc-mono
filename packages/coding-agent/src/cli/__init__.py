"""CLI entry point for sentarc-coding-agent."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point — parse args and dispatch to appropriate mode."""
    if argv is None:
        argv = sys.argv[1:]
    try:
        asyncio.run(_main_async(argv))
    except KeyboardInterrupt:
        sys.exit(0)


async def _main_async(argv: List[str]) -> None:
    from sentarc_coding_agent.cli.args import parse_args, print_help
    from sentarc_coding_agent.config import VERSION, APP_NAME, get_agent_dir, get_custom_themes_dir
    from sentarc_coding_agent.core.tools import create_tools, TOOL_NAMES
    from sentarc_coding_agent.core.system_prompt import build_system_prompt
    from sentarc_coding_agent.core.model_resolver import resolve_model
    from sentarc_coding_agent.core.settings_manager import load_settings
    from sentarc_coding_agent.core.skills import load_skills
    from sentarc_coding_agent.core.extensions.loader import discover_extensions
    from sentarc_coding_agent.core.messages import convert_to_llm

    args = parse_args(argv)

    # --- Simple flag handling ---
    if args.get("help"):
        print_help()
        return

    if args.get("version"):
        print(f"{APP_NAME} {VERSION}")
        return

    if args.get("list_models") is not None:
        from sentarc_coding_agent.cli.list_models import list_models
        search = args["list_models"] if isinstance(args["list_models"], str) else None
        list_models(search)
        return

    # --- Load settings ---
    settings = load_settings()

    # --- Resolve model ---
    provider = args.get("provider") or settings.provider or "google"
    model_spec = args.get("model") or settings.model or "gemini-2.5-flash"
    resolved_provider, model_id, model_thinking = resolve_model(provider, model_spec)
    thinking_level = args.get("thinking") or model_thinking or settings.thinking or "off"

    try:
        from sentarc_ai.models import get_model
        model_def = get_model(resolved_provider, model_id)
    except Exception as e:
        print(f"Error: Could not resolve model {resolved_provider}/{model_id}: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Build tools ---
    cwd = os.getcwd()
    no_tools = args.get("no_tools", False)
    if no_tools:
        tool_names: List[str] = []
    elif args.get("tools"):
        tool_names = args["tools"]
    else:
        tool_names = list(settings.tools or ["read", "bash", "edit", "write"])

    tools = create_tools(cwd, tool_names) if tool_names else []

    # --- Load skills ---
    skills: list = []
    if not args.get("no_skills"):
        try:
            skill_paths = args.get("skills") or []
            loaded_skills, _ = load_skills(cwd=cwd, skill_paths=skill_paths)
            skills = loaded_skills
        except Exception:
            pass

    # --- Load context files ---
    context_files: list = []
    if args.get("file_args"):
        try:
            from sentarc_coding_agent.cli.file_processor import load_file_args
            context_files = [
                f for f in load_file_args(args["file_args"], cwd)
                if f.get("type") == "text"
            ]
        except Exception as e:
            print(f"Warning: Could not load context files: {e}", file=sys.stderr)

    # --- Build system prompt ---
    custom_prompt = args.get("system_prompt")
    append_prompt = args.get("append_system_prompt")
    system_prompt = build_system_prompt(
        custom_prompt=custom_prompt,
        selected_tools=None if no_tools else tool_names,
        append_system_prompt=append_prompt,
        cwd=cwd,
        context_files=context_files if context_files else None,
        skills=skills if skills else None,
    )

    # --- Load extensions ---
    all_agent_tools = list(tools)
    if not args.get("no_extensions"):
        try:
            agent_dir = get_agent_dir()
            extension_paths = args.get("extensions") or []
            extensions = discover_extensions(
                extension_paths=extension_paths,
                agent_dir=agent_dir,
            )
            for ext in extensions:
                custom = getattr(ext, "custom_tools", None)
                if custom:
                    all_agent_tools.extend(custom)
        except Exception:
            pass

    # --- Create agent ---
    try:
        from sentarc_agent.agent import Agent
        from sentarc_agent.types import AgentOptions
    except ImportError as e:
        print(f"Error: sentarc-agent not available: {e}", file=sys.stderr)
        sys.exit(1)

    agent = Agent(AgentOptions(
        convert_to_llm=convert_to_llm,
        initial_state={
            "system_prompt": system_prompt,
            "model": model_def,
            "thinking_level": thinking_level,
            "tools": all_agent_tools,
        },
    ))

    # --- Set up session ---
    from sentarc_coding_agent.core.session_manager import SessionManager

    session: Optional[SessionManager] = None
    no_session = args.get("no_session", False)
    if not no_session:
        session_file = args.get("session")
        if args.get("resume"):
            from sentarc_coding_agent.cli.session_picker import select_session_interactive
            session_id = select_session_interactive()
            if session_id:
                try:
                    session = SessionManager.open(session_id)
                except Exception:
                    session = SessionManager.create(cwd)
            else:
                session = SessionManager.create(cwd)
        elif args.get("continue"):
            session = SessionManager.continue_recent(cwd)
        elif session_file:
            session = SessionManager.open(session_file)
        else:
            session = SessionManager.create(cwd)

    # Load session history into agent state
    if session:
        context = session.build_session_context()
        history = context.get("messages", [])
        if history:
            agent._state.messages = history
        session_thinking = context.get("thinkingLevel")
        if session_thinking and not args.get("thinking"):
            agent._state.thinking_level = session_thinking

    # --- Determine mode and run ---
    mode = args.get("mode")
    messages: List[str] = args.get("messages", [])
    has_messages = bool(messages)
    is_print = args.get("print", False) or has_messages

    if mode == "rpc":
        from sentarc_coding_agent.modes.rpc.rpc_mode import run_rpc_mode
        await run_rpc_mode(agent)

    elif is_print or mode == "json":
        from sentarc_coding_agent.modes.print_mode import PrintModeOptions, run_print_mode
        initial_msg = messages[0] if messages else None
        extra_msgs = messages[1:] if len(messages) > 1 else []
        opts = PrintModeOptions(
            mode=mode if mode in ("text", "json") else "text",
            initial_message=initial_msg,
            messages=extra_msgs,
        )
        await run_print_mode(agent, opts)

    else:
        # Interactive TUI mode
        try:
            from sentarc_coding_agent.modes.interactive.interactive_mode import run_interactive_mode
            from sentarc_coding_agent.modes.interactive.theme import load_theme
            from pathlib import Path

            themes_dir = Path(get_custom_themes_dir())
            theme = load_theme("dark", themes_dir)
            await run_interactive_mode(agent, theme=theme)
        except ImportError:
            # Fallback: basic stdin loop
            from sentarc_coding_agent.core.agent_session import run_agent_session
            await run_agent_session(args)


if __name__ == "__main__":
    main()
