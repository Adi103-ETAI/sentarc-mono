"""
Print mode: send prompts, output result, exit.
Used for: arc -p "prompt" (text output) or arc --mode json "prompt" (JSON event stream)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class PrintModeOptions:
    mode: Literal["text", "json"] = "text"
    messages: List[str] = field(default_factory=list)
    initial_message: Optional[str] = None
    initial_images: Optional[list] = None


async def run_print_mode(agent_session, options: PrintModeOptions) -> None:
    """
    Send prompts to the agent and output results.
    - text mode: output final assistant text to stdout
    - json mode: output all AgentEvents as JSON lines to stdout
    """
    agent = agent_session if hasattr(agent_session, "subscribe") else agent_session.agent

    if options.mode == "json":
        agent.subscribe(_make_json_handler())

    if options.initial_message:
        await agent.prompt(options.initial_message)

    for msg in (options.messages or []):
        await agent.prompt(msg)

    if options.mode == "text":
        _print_last_assistant_text(agent)

    sys.stdout.flush()


def _make_json_handler():
    def handle_event(event):
        if hasattr(event, "__dict__"):
            d = {k: v for k, v in event.__dict__.items() if not callable(v)}
        else:
            d = {"event": str(event)}
        print(json.dumps(d, default=str), flush=True)
    return handle_event


def _print_last_assistant_text(agent) -> None:
    messages = agent._state.messages
    if not messages:
        return
    last = messages[-1]
    role = last.get("role") if isinstance(last, dict) else getattr(last, "role", None)
    if role != "assistant":
        return
    content = last.get("content", []) if isinstance(last, dict) else getattr(last, "content", [])
    for block in (content or []):
        btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if btype == "text":
            text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
            if text:
                print(text)
