"""
AgentSession — wraps sentarc-agent's Agent with session persistence.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict, List, Optional

from sentarc_coding_agent.core.messages import convert_to_llm
from sentarc_coding_agent.core.model_resolver import resolve_model
from sentarc_coding_agent.core.session_manager import SessionManager
from sentarc_coding_agent.core.settings_manager import load_settings
from sentarc_coding_agent.core.system_prompt import build_system_prompt
from sentarc_coding_agent.core.tools import create_tools
from sentarc_coding_agent.core.event_recorder import attach_event_recorder, create_event_log_path
from sentarc_coding_agent.config import get_agent_dir


async def run_agent_session(args: Dict[str, Any]) -> None:
    """
    Run an agent session based on parsed CLI args.
    Handles both interactive and print (non-interactive) modes.
    """
    cwd = os.getcwd()
    settings = load_settings(cwd)

    provider = args.get("provider") or settings.provider or "google"
    model_spec = args.get("model") or settings.model or "gemini-2.5-flash"
    api_key = args.get("api_key")
    thinking_level = args.get("thinking") or settings.thinking or "off"
    print_mode = args.get("print", False)
    no_session = args.get("no_session", False)
    no_tools = args.get("no_tools", False)
    tool_names = args.get("tools") or settings.tools or ["read", "bash", "edit", "write"]
    messages = args.get("messages", [])
    session_file = args.get("session")
    session_dir = args.get("session_dir")

    # Resolve model
    resolved_provider, model_id, model_thinking = resolve_model(provider, model_spec)
    if model_thinking and not args.get("thinking"):
        thinking_level = model_thinking

    # Build tools
    bash_security_profile = args.get("bash_security_profile") or settings.bash_security_profile or "standard"
    bash_block_patterns = list(settings.bash_block_patterns or [])
    bash_options = {
        "security_profile": bash_security_profile,
        "blocked_patterns": bash_block_patterns,
    }
    tools = [] if no_tools else create_tools(cwd, tool_names, bash_options=bash_options)

    # Build system prompt
    custom_prompt = args.get("system_prompt")
    append_prompt = args.get("append_system_prompt")
    system_prompt = build_system_prompt(
        custom_prompt=custom_prompt,
        selected_tools=None if no_tools else tool_names,
        append_system_prompt=append_prompt,
        cwd=cwd,
    )

    # Set up session
    session: Optional[SessionManager] = None
    if not no_session:
        if session_file:
            session = SessionManager.open(session_file, session_dir)
        elif args.get("continue"):
            session = SessionManager.continue_recent(cwd, session_dir)
        else:
            session = SessionManager.create(cwd, session_dir)

    # Load session context
    context = session.build_session_context() if session else {"messages": [], "thinkingLevel": "off", "model": None}
    history = context.get("messages", [])

    # Build initial user message
    initial_message: Optional[str] = " ".join(messages) if messages else None

    try:
        from sentarc_ai.models import get_model
        model_def = get_model(resolved_provider, model_id)
    except Exception as e:
        print(f"Error: Could not resolve model {resolved_provider}/{model_id}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        from sentarc_agent.agent import Agent
        from sentarc_agent.types import AgentOptions
    except ImportError as e:
        print(f"Error: sentarc-agent not available: {e}", file=sys.stderr)
        sys.exit(1)

    agent = Agent(AgentOptions(
        convert_to_llm=convert_to_llm,
        get_api_key=(lambda _provider: api_key) if api_key else None,
        initial_state={
            "system_prompt": system_prompt,
            "model": model_def,
            "thinking_level": thinking_level,
            "tools": tools,
            "messages": history,
        },
    ))

    event_log_enabled = bool(args.get("event_log") or settings.event_log_enabled)
    if event_log_enabled:
        event_log_path = args.get("event_log_path") or settings.event_log_path or create_event_log_path(get_agent_dir())
        attach_event_recorder(
            agent,
            event_log_path,
            metadata={
                "provider": resolved_provider,
                "model": model_id,
                "mode": "fallback-interactive" if not print_mode else "fallback-print",
                "cwd": cwd,
            },
        )

    def on_event(event: Any) -> None:
        """Handle agent events for display."""
        etype = getattr(event, "type", "")
        if etype == "message_update":
            # Stream text output
            msg = getattr(event, "message", None)
            if msg:
                role = getattr(msg, "role", "") or msg.get("role", "") if isinstance(msg, dict) else ""
                if role == "assistant":
                    content = getattr(msg, "content", []) or msg.get("content", []) if isinstance(msg, dict) else []
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "")
                                if text:
                                    print(text, end="", flush=True)
        elif etype == "agent_end":
            print()  # Final newline

    agent.subscribe(on_event)

    if initial_message:
        user_msg = {
            "role": "user",
            "content": [{"type": "text", "text": initial_message}],
        }

        if session:
            session.append_message(user_msg)

        async def _run():
            before_count = len(agent.state.messages)
            await agent.prompt(user_msg)
            if session:
                for msg in agent.state.messages[before_count:]:
                    role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
                    if role in ("assistant", "tool", "toolResult"):
                        session.append_message(msg if isinstance(msg, dict) else vars(msg))

        await _run()

    elif not print_mode:
        # Interactive mode - basic stdin loop
        print(f"arc — interactive mode (Ctrl+C to exit)")
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue

                user_msg = {
                    "role": "user",
                    "content": [{"type": "text", "text": user_input}],
                }
                if session:
                    session.append_message(user_msg)

                before_count = len(agent.state.messages)
                await agent.prompt(user_msg)

                if session:
                    for msg in agent.state.messages[before_count:]:
                        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
                        if role in ("assistant", "tool", "toolResult"):
                            session.append_message(msg if isinstance(msg, dict) else vars(msg))

            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
