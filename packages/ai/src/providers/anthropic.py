"""
Anthropic provider.
Python port of packages/ai/src/providers/anthropic.ts from sentarc-mono.
"""
from __future__ import annotations
import json
from typing import AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import Context, ModelDef, StreamEvent

from ..types import (
    Message, Role,
    ToolCallContent, ToolResultContent,
    TextDeltaEvent, ThinkingDeltaEvent,
    ToolCallStartEvent, ToolCallDeltaEvent, ToolCallEndEvent,
    MessageStartEvent, MessageEndEvent, ErrorEvent,
    TokenUsage,
)


def _fmt_messages(messages: list[Message]) -> list[dict]:
    """Convert sentarc-ai Messages → Anthropic API format."""
    result = []
    for m in messages:
        if m.role == Role.SYSTEM:
            continue  # system prompt is a top-level param in Anthropic API
        if m.tool_calls:
            content = []
            if isinstance(m.content, str) and m.content:
                content.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                content.append({
                    "type":  "tool_use",
                    "id":    tc.id,
                    "name":  tc.name,
                    "input": tc.arguments,
                })
            result.append({"role": "assistant", "content": content})
        elif m.tool_call_id:
            result.append({"role": "user", "content": [{
                "type":        "tool_result",
                "tool_use_id": m.tool_call_id,
                "content":     m.content if isinstance(m.content, str) else "",
            }]})
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
    thinking: bool = False,
) -> AsyncIterator["StreamEvent"]:
    """Stream from Anthropic API, yielding normalized StreamEvents."""

    system = context.system_prompt
    messages = _fmt_messages(context.messages)

    params: dict = {
        "model":      model.id,
        "max_tokens": model.max_output,
        "messages":   messages,
    }
    if system:
        params["system"] = system
    if context.tools:
        params["tools"] = [t.to_anthropic() for t in context.tools]
    if thinking and model.supports_thinking:
        params["thinking"] = {"type": "enabled", "budget_tokens": 5_000}

    yield MessageStartEvent()
    building: dict[str, dict] = {}

    try:
        async with client.messages.stream(**params) as s:
            async for ev in s:
                t = ev.type

                if t == "content_block_start":
                    cb = ev.content_block
                    if cb.type == "tool_use":
                        building[cb.id] = {"id": cb.id, "name": cb.name, "raw": ""}
                        yield ToolCallStartEvent(id=cb.id, name=cb.name)

                elif t == "content_block_delta":
                    d = ev.delta
                    if d.type == "text_delta":
                        yield TextDeltaEvent(text=d.text)
                    elif d.type == "thinking_delta":
                        yield ThinkingDeltaEvent(thinking=d.thinking)
                    elif d.type == "input_json_delta":
                        for tc in building.values():
                            tc["raw"] += d.partial_json
                            yield ToolCallDeltaEvent(id=tc["id"], partial_input=d.partial_json)

                elif t == "message_stop":
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

                    fm = await s.get_final_message()
                    u  = fm.usage
                    yield MessageEndEvent(usage=TokenUsage(
                        input_tokens       = u.input_tokens,
                        output_tokens      = u.output_tokens,
                        cache_read_tokens  = getattr(u, "cache_read_input_tokens", 0),
                        cache_write_tokens = getattr(u, "cache_creation_input_tokens", 0),
                    ))

    except Exception as e:
        yield ErrorEvent(error=str(e))
