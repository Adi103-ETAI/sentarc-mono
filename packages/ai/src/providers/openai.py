"""
OpenAI-compatible provider (also used for Ollama).
Python port of packages/ai/src/providers/openai-completions.ts from sentarc-mono.
"""
from __future__ import annotations
import json
from typing import AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import Context, ModelDef, StreamEvent

from ..types import (
    Message, Role,
    ToolCallContent,
    TextDeltaEvent,
    ToolCallStartEvent, ToolCallDeltaEvent, ToolCallEndEvent,
    MessageStartEvent, MessageEndEvent, ErrorEvent,
    TokenUsage,
)


def _fmt_messages(messages: list[Message]) -> list[dict]:
    """Convert sentarc-ai Messages → OpenAI API format."""
    result = []
    for m in messages:
        if m.role == Role.SYSTEM:
            result.append({"role": "system", "content": m.content})
        elif m.tool_call_id:
            result.append({
                "role":         "tool",
                "tool_call_id": m.tool_call_id,
                "content":      m.content if isinstance(m.content, str) else "",
            })
        elif m.tool_calls:
            result.append({
                "role":       "assistant",
                "content":    m.content or None,
                "tool_calls": [{
                    "id":   tc.id,
                    "type": "function",
                    "function": {
                        "name":      tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                } for tc in m.tool_calls],
            })
        else:
            result.append({
                "role":    m.role.value,
                "content": m.content if isinstance(m.content, str) else "",
            })
    return result


async def stream(
    client,
    model: "ModelDef",
    context: "Context",
) -> AsyncIterator["StreamEvent"]:
    """Stream from OpenAI-compatible API, yielding normalized StreamEvents."""

    messages = _fmt_messages(context.messages)

    # Inject system prompt as first message if present and not already there
    if context.system_prompt:
        messages = [{"role": "system", "content": context.system_prompt}] + messages

    params: dict = {
        "model":          model.id,
        "max_tokens":     model.max_output,
        "messages":       messages,
        "stream":         True,
        "stream_options": {"include_usage": True},
    }
    if context.tools:
        params["tools"] = [t.to_openai() for t in context.tools]

    yield MessageStartEvent()
    building: dict[int, dict] = {}

    try:
        async for chunk in await client.chat.completions.create(**params):
            # Usage-only chunk (no choices)
            if not chunk.choices:
                if chunk.usage:
                    yield MessageEndEvent(usage=TokenUsage(
                        input_tokens  = chunk.usage.prompt_tokens,
                        output_tokens = chunk.usage.completion_tokens,
                    ))
                continue

            delta  = chunk.choices[0].delta
            finish = chunk.choices[0].finish_reason

            if delta.content:
                yield TextDeltaEvent(text=delta.content)

            if delta.tool_calls:
                for tcd in delta.tool_calls:
                    idx = tcd.index
                    if idx not in building:
                        name = tcd.function.name or ""
                        building[idx] = {"id": tcd.id or "", "name": name, "raw": ""}
                        yield ToolCallStartEvent(id=building[idx]["id"], name=name)
                    if tcd.function.arguments:
                        building[idx]["raw"] += tcd.function.arguments
                        yield ToolCallDeltaEvent(
                            id=building[idx]["id"],
                            partial_input=tcd.function.arguments,
                        )

            if finish in ("tool_calls", "stop"):
                for tc in building.values():
                    try:
                        args = json.loads(tc["raw"]) if tc["raw"] else {}
                    except Exception:
                        args = {}
                    yield ToolCallEndEvent(
                        tool_call=ToolCallContent(
                            id=tc["id"], name=tc["name"], arguments=args
                        )
                    )
                building.clear()
                if finish == "stop":
                    yield MessageEndEvent(usage=TokenUsage())

    except Exception as e:
        yield ErrorEvent(error=str(e))
