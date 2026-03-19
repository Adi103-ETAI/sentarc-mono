"""
Anthropic provider.
"""
from __future__ import annotations
import json
from typing import AsyncIterator, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from ..types import Context, ModelDef

from ..types import (
    ToolUseContent, TokenUsage, ReasoningEffort, StreamEvent,
    StartEvent, TextStartEvent, TextDeltaEvent, TextEndEvent,
    ThinkingStartEvent, ThinkingDeltaEvent, ThinkingEndEvent,
    ToolUseStartEvent, ToolUseDeltaEvent, ToolUseEndEvent,
    StopEvent, ErrorEvent,
)
from ..transform_messages import to_anthropic


def _map_reasoning(effort: ReasoningEffort) -> dict | None:
    """Map ReasoningEffort → Anthropic thinking param."""
    if effort == ReasoningEffort.NONE:
        return None
    # xhigh clamped to high for Anthropic
    budget = {
        ReasoningEffort.LOW:    1_000,
        ReasoningEffort.MEDIUM: 5_000,
        ReasoningEffort.HIGH:   10_000,
        ReasoningEffort.XHIGH:  10_000,
    }.get(effort, 5_000)
    return {"type": "enabled", "budget_tokens": budget}



class AnthropicProvider:
    def __init__(self):
        self.api = "anthropic"

    async def stream(
        self,
        model: "ModelDef",
        context: "Context",
        options: Optional[any] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream from Anthropic API."""

        from anthropic import AsyncAnthropic
        from ..env import get_env_api_key, sanitize_api_key
        from ..transform_messages import to_anthropic, transform_messages

        api_key = get_env_api_key("anthropic")
        if not api_key:
            raise RuntimeError("No API key found for anthropic")

        try:
            client_instance = AsyncAnthropic(api_key=api_key, base_url=model.base_url)
        except Exception as e:
            # Sanitize API key in error messages
            safe_key = sanitize_api_key(api_key)
            raise RuntimeError(f"Failed to initialize Anthropic client (key: {safe_key}): {e}") from e
        
        reasoning = options.reasoning_effort if options else ReasoningEffort.NONE

        # 1. Transform messages
        transformed_messages = transform_messages(context.messages, model)

        params: dict = {
            "model":      model.id,
            "max_tokens": model.max_output,
            "messages":   to_anthropic(transformed_messages),
        }
        if context.system_prompt:
            params["system"] = context.system_prompt
        if context.tools:
            params["tools"] = [t.to_anthropic() for t in context.tools]
        thinking = _map_reasoning(reasoning)
        if thinking:
            params["thinking"] = thinking

        yield StartEvent(model=model.id)
        building: dict[str, dict] = {}
        text_buf  = ""
        think_buf = ""
        in_text   = False
        in_think  = False

        try:
            async with client_instance.messages.stream(**params) as s:
                async for ev in s:
                    t = ev.type

                    if t == "content_block_start":
                        cb = ev.content_block
                        if cb.type == "text":
                            in_text = True
                            yield TextStartEvent()
                        elif cb.type == "thinking":
                            in_think = True
                            yield ThinkingStartEvent()
                        elif cb.type == "tool_use":
                            building[cb.id] = {"id": cb.id, "name": cb.name, "raw": ""}
                            yield ToolUseStartEvent(id=cb.id, name=cb.name)

                    elif t == "content_block_delta":
                        d = ev.delta
                        if d.type == "text_delta":
                            text_buf += d.text
                            yield TextDeltaEvent(text=d.text)
                        elif d.type == "thinking_delta":
                            think_buf += d.thinking
                            yield ThinkingDeltaEvent(thinking=d.thinking)
                        elif d.type == "input_json_delta":
                            for tc in building.values():
                                tc["raw"] += d.partial_json
                                yield ToolUseDeltaEvent(id=tc["id"], partial_input=d.partial_json)

                    elif t == "content_block_stop":
                        if in_text:
                            yield TextEndEvent(text=text_buf)
                            text_buf = ""
                            in_text  = False
                        elif in_think:
                            yield ThinkingEndEvent(thinking=think_buf)
                            think_buf = ""
                            in_think  = False

                    elif t == "message_stop":
                        for tc in building.values():
                            try:
                                args = json.loads(tc["raw"]) if tc["raw"] else {}
                            except Exception:
                                args = {}
                            yield ToolUseEndEvent(
                                tool_use=ToolUseContent(id=tc["id"], name=tc["name"], arguments=args)
                            )

                        fm = await s.get_final_message()
                        u  = fm.usage
                        yield StopEvent(
                            stop_reason=fm.stop_reason or "end_turn",
                            usage=TokenUsage(
                                input_tokens       = u.input_tokens,
                                output_tokens      = u.output_tokens,
                                cache_read_tokens  = getattr(u, "cache_read_input_tokens", 0),
                                cache_write_tokens = getattr(u, "cache_creation_input_tokens", 0),
                            ),
                        )

        except Exception as e:
            raise RuntimeError(str(e))
