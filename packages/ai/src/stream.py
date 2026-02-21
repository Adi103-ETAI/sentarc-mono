"""
Unified streaming entry point.

Usage:
    async for event in stream(model, context):
        ...
"""
from __future__ import annotations
from typing import AsyncIterator, Optional, Tuple, Literal

from .types import Context, ModelDef, StreamEvent, StreamOptions, ReasoningEffort, ErrorEvent
from .registry import get_api_provider

async def stream(
    model: ModelDef,
    context: Context,
    options: Optional[StreamOptions] = None,
) -> AsyncIterator[StreamEvent]:
    """
    Unified streaming interface across all providers.
    Yields StreamEvent objects normalised to a common format.
    """
    provider = get_api_provider(model.api)
    if not provider:
        yield ErrorEvent(error=f"No provider implementation for api={model.api!r}")
        return
        
    async for event in provider.stream(model, context, options):
        yield event


async def complete(
    model: ModelDef,
    context: Context,
    options: Optional[StreamOptions] = None,
) -> tuple[str, list]:
    """
    Non-streaming helper. Returns (text, tool_calls).
    Useful for simple one-shot calls.
    """
    from .types import TextEndEvent, ToolUseEndEvent

    final_text = ""
    tool_calls = []

    async for event in stream(model, context, options):
        if isinstance(event, TextEndEvent):
            final_text = event.text
        elif isinstance(event, ToolUseEndEvent):
            tool_calls.append(event.tool_use)

    return final_text, tool_calls


async def stream_simple(
    model: ModelDef,
    context: Context,
    reasoning: Literal["none", "low", "medium", "high", "xhigh"] = "none",
) -> AsyncIterator[StreamEvent]:
    """
    Unified Interface for reasoning models. Maps a simple reasoning effort string
    to the appropriate StreamOptions structure to seamlessly support thinking
    across different API providers (Anthropic, OpenAI, etc).
    """
    effort = ReasoningEffort(reasoning) if reasoning != "none" else ReasoningEffort.NONE
    
    # Map simple unified intent into provider-agnostic StreamOptions wrapper
    options = StreamOptions(
        reasoning_effort=effort,
        thinking_enabled=(effort != ReasoningEffort.NONE),
        thinking_budget=10_000 if effort != ReasoningEffort.NONE else 0
    )
    
    provider = get_api_provider(model.api)
    if not provider:
        yield ErrorEvent(error=f"No provider implementation for api={model.api!r}")
        return
        
    # Allow providers to have explicit simple streaming handlers if configured, 
    # otherwise fallback to generic stream wrapper with injected options
    if hasattr(provider, "stream_simple") and provider.stream_simple != provider.stream:
        async for event in provider.stream_simple(model, context, options):
            yield event
    else:
        async for event in provider.stream(model, context, options):
            yield event


async def complete_simple(
    model: ModelDef,
    context: Context,
    reasoning: Literal["none", "low", "medium", "high", "xhigh"] = "none",
) -> tuple[str, list]:
    """
    Non-streaming helper for unified reasoning. Returns (text, tool_calls).
    """
    from .types import TextEndEvent, ToolUseEndEvent

    final_text = ""
    tool_calls = []

    async for event in stream_simple(model, context, reasoning):
        if isinstance(event, TextEndEvent):
            final_text = event.text
        elif isinstance(event, ToolUseEndEvent):
            tool_calls.append(event.tool_use)

    return final_text, tool_calls
