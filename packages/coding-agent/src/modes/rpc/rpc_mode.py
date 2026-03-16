"""
RPC mode: JSON-RPC server over stdin/stdout.
Each command is a JSON line on stdin. Each response/event is a JSON line on stdout.
"""
from __future__ import annotations

import asyncio
import json
import sys

from sentarc_coding_agent.modes.rpc.jsonl import write_jsonl, JsonlReader
from sentarc_coding_agent.modes.rpc.rpc_types import make_response


async def run_rpc_mode(agent_session) -> None:
    """
    Read JSON commands from stdin, dispatch to agent, write JSON responses to stdout.
    """
    agent = agent_session if hasattr(agent_session, "subscribe") else agent_session.agent

    # Forward all agent events to stdout
    def forward_event(event):
        if hasattr(event, "__dict__"):
            d = {k: v for k, v in event.__dict__.items() if not callable(v)}
            write_jsonl({"type": "event", **d})
        else:
            write_jsonl({"type": "event", "event": str(event)})

    agent.subscribe(forward_event)

    reader = JsonlReader()
    async for cmd in reader:
        cmd_type = cmd.get("type")
        cmd_id = cmd.get("id")

        try:
            if cmd_type == "prompt":
                message = cmd.get("message", "")
                asyncio.create_task(agent.prompt(message))
                write_jsonl(make_response("prompt", True, id=cmd_id))

            elif cmd_type == "steer":
                message = cmd.get("message", "")
                asyncio.create_task(agent.steer(message) if hasattr(agent, "steer") else agent.prompt(message))
                write_jsonl(make_response("steer", True, id=cmd_id))

            elif cmd_type == "abort":
                if hasattr(agent, "abort"):
                    agent.abort()
                write_jsonl(make_response("abort", True, id=cmd_id))

            elif cmd_type == "get_state":
                state = agent._state
                data = {
                    "thinking_level": state.thinking_level,
                    "is_streaming": state.is_streaming,
                    "message_count": len(state.messages),
                    "session_id": getattr(agent, "session_id", None),
                }
                if state.model:
                    model = state.model
                    data["model"] = {
                        "id": getattr(model, "id", str(model)),
                        "provider": getattr(model, "provider", ""),
                    }
                write_jsonl(make_response("get_state", True, data=data, id=cmd_id))

            elif cmd_type == "get_messages":
                messages = agent._state.messages
                serializable = []
                for m in messages:
                    if isinstance(m, dict):
                        serializable.append(m)
                    elif hasattr(m, "__dict__"):
                        serializable.append({k: v for k, v in m.__dict__.items() if not callable(v)})
                write_jsonl(make_response("get_messages", True, data={"messages": serializable}, id=cmd_id))

            elif cmd_type == "set_model":
                provider = cmd.get("provider", "")
                model_id = cmd.get("model_id") or cmd.get("modelId", "")
                from sentarc_ai.models import get_model
                model = get_model(provider, model_id)
                agent.set_model(model)
                write_jsonl(make_response("set_model", True, data={
                    "id": getattr(model, "id", model_id),
                    "provider": getattr(model, "provider", provider),
                }, id=cmd_id))

            elif cmd_type == "set_thinking_level":
                level = cmd.get("level", "off")
                agent.set_thinking_level(level)
                write_jsonl(make_response("set_thinking_level", True, id=cmd_id))

            elif cmd_type == "get_last_assistant_text":
                messages = agent._state.messages
                text = ""
                for m in reversed(messages):
                    role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                    if role == "assistant":
                        content = m.get("content", []) if isinstance(m, dict) else getattr(m, "content", [])
                        for block in (content or []):
                            btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                            if btype == "text":
                                text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
                        break
                write_jsonl(make_response("get_last_assistant_text", True, data={"text": text}, id=cmd_id))

            elif cmd_type == "new_session":
                write_jsonl(make_response("new_session", True, data={"cancelled": False}, id=cmd_id))

            elif cmd_type == "get_available_models":
                try:
                    from sentarc_ai.models import list_models
                    models = list_models()
                    data = {"models": [
                        {"id": getattr(m, "id", str(m)), "provider": getattr(m, "provider", "")}
                        for m in models
                    ]}
                except Exception:
                    data = {"models": []}
                write_jsonl(make_response("get_available_models", True, data=data, id=cmd_id))

            else:
                write_jsonl(make_response(
                    cmd_type or "unknown", False,
                    error=f"Unknown command: {cmd_type}",
                    id=cmd_id,
                ))

        except Exception as e:
            write_jsonl(make_response(cmd_type or "unknown", False, error=str(e), id=cmd_id))
