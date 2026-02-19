"""
Unified streaming entry point.
Python port of packages/ai/src/stream.ts from sentarc-mono.

Usage:
    async for event in stream(model, context):
        ...
"""
from __future__ import annotations
import os
from typing import AsyncIterator

from .types import Context, ModelDef, StreamEvent
from .models import resolve_model


def _build_anthropic_client(model: ModelDef):
    import anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    return anthropic.AsyncAnthropic(api_key=api_key)


def _build_openai_client(model: ModelDef):
    import openai
    if model.provider == "ollama":
        return openai.AsyncOpenAI(
            api_key="ollama",
            base_url=model.base_url or "http://localhost:11434/v1",
        )
    return openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


async def stream(
    model:    ModelDef,
    context:  Context,
    thinking: bool = False,
) -> AsyncIterator[StreamEvent]:
    """
    Unified streaming interface across all providers.
    Yields StreamEvent objects normalised to a common format.
    """
    if model.api == "anthropic":
        from .providers.anthropic import stream as _stream
        client = _build_anthropic_client(model)
        async for event in _stream(client, model, context, thinking=thinking):
            yield event

    elif model.api == "openai":
        from .providers.openai import stream as _stream
        client = _build_openai_client(model)
        async for event in _stream(client, model, context):
            yield event

    else:
        from .types import ErrorEvent
        yield ErrorEvent(error=f"No provider implementation for api={model.api!r}")


async def complete(
    model:   ModelDef,
    context: Context,
) -> tuple[str, list]:
    """
    Non-streaming helper. Returns (text, tool_calls).
    Useful for simple one-shot calls.
    """
    from .types import TextDeltaEvent, ToolCallEndEvent

    text_parts = []
    tool_calls = []

    async for event in stream(model, context):
        if isinstance(event, TextDeltaEvent):
            text_parts.append(event.text)
        elif isinstance(event, ToolCallEndEvent):
            tool_calls.append(event.tool_call)

    return "".join(text_parts), tool_calls
