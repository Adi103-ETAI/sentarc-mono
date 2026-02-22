"""
Unified streaming entry point.

Usage:
    async for event in stream(model, context):
        ...
"""
from __future__ import annotations
from typing import AsyncIterator, Optional, Tuple, Literal

from .types import Context, ModelDef, StreamEvent, StreamOptions, ReasoningEffort, ErrorEvent, AssistantMessage, TextContent, ThinkingContent, ToolUseContent, TokenUsage
from .registry import get_api_provider

async def _with_error_handling(
    generator: AsyncIterator[StreamEvent],
    model: ModelDef
) -> AsyncIterator[StreamEvent]:
    import asyncio
    import json
    from .types import TextContent, ThinkingContent, ToolUseContent, TokenUsage, AssistantMessage, ErrorEvent
    
    blocks = []
    current_text = ""
    current_thinking = ""
    current_tool_id = ""
    current_tool_name = ""
    current_tool_args = ""
    usage = TokenUsage()

    try:
        async for event in generator:
            if event.type == "text_delta":
                current_text += event.text
            elif event.type == "text_end":
                if current_text:
                    blocks.append(TextContent(text=current_text))
                    current_text = ""
            elif event.type == "thinking_delta":
                current_thinking += event.thinking
            elif event.type == "thinking_end":
                if current_thinking:
                    blocks.append(ThinkingContent(thinking=current_thinking))
                    current_thinking = ""
            elif event.type == "tool_use_start":
                current_tool_id = event.id
                current_tool_name = event.name
                current_tool_args = ""
            elif event.type == "tool_use_delta":
                current_tool_args += event.partial_input
            elif event.type == "tool_use_end":
                blocks.append(event.tool_use)
                current_tool_id = ""
                current_tool_name = ""
                current_tool_args = ""
            elif event.type == "stop":
                usage = event.usage

            yield event
            
    except asyncio.CancelledError:
        content = list(blocks)
        if current_text: content.append(TextContent(text=current_text))
        if current_thinking: content.append(ThinkingContent(thinking=current_thinking))
        
        msg = AssistantMessage(
            role="assistant",
            content=content,
            stop_reason="aborted",
            usage=usage,
            provider=model.provider,
            model=model.id,
            api=model.api,
            error_message="Request was cancelled via abort/cancellation signal"
        )
        yield ErrorEvent(error=msg, reason="aborted")
        
    except Exception as e:
        content = list(blocks)
        if current_text: content.append(TextContent(text=current_text))
        if current_thinking: content.append(ThinkingContent(thinking=current_thinking))
        
        msg = AssistantMessage(
            role="assistant",
            content=content,
            stop_reason="error",
            usage=usage,
            provider=model.provider,
            model=model.id,
            api=model.api,
            error_message=str(e)
        )
        yield ErrorEvent(error=msg, reason="error")

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
        msg = AssistantMessage(
            role="assistant", content=[], stop_reason="error", 
            provider=model.provider, model=model.id, api=model.api,
            error_message=f"No provider implementation for api={model.api!r}"
        )
        yield ErrorEvent(error=msg, reason="error")
        return
        
    async for event in _with_error_handling(provider.stream(model, context, options), model):
        yield event


async def complete(
    model: ModelDef,
    context: Context,
    options: Optional[StreamOptions] = None,
) -> AssistantMessage:
    """
    Non-streaming helper. Returns the final AssistantMessage.
    Useful for simple one-shot calls.
    """
    from .types import AssistantMessage, TextContent, ThinkingContent, ToolUseContent, TokenUsage, StopEvent, ErrorEvent

    blocks = []
    usage = TokenUsage()
    stop_reason = "stop"
    error_message = None

    async for event in stream(model, context, options):
        if event.type == "text_end":
            blocks.append(TextContent(text=event.text))
        elif event.type == "thinking_end":
            blocks.append(ThinkingContent(thinking=event.thinking))
        elif event.type == "tool_use_end":
            blocks.append(event.tool_use)
        elif event.type == "stop":
            usage = event.usage
            stop_reason = event.stop_reason
        elif event.type == "error":
            usage = event.error.usage
            blocks = event.error.content
            stop_reason = event.reason
            error_message = event.error.error_message
            break

    return AssistantMessage(
        role="assistant",
        content=blocks,
        stop_reason=stop_reason,
        usage=usage,
        provider=model.provider,
        model=model.id,
        api=model.api,
        error_message=error_message
    )


async def stream_simple(
    model: ModelDef,
    context: Context,
    options: Optional[StreamOptions] = None,
) -> AsyncIterator[StreamEvent]:
    """
    Unified Interface for reasoning models and standard streams.
    """
    options = options or StreamOptions()

    
    provider = get_api_provider(model.api)
    if not provider:
        yield ErrorEvent(error=AssistantMessage(
            role="assistant", content=[], stop_reason="error", provider=model.provider, model=model.id, api=model.api, error_message=f"No provider implementation for api={model.api!r}"
        ), reason="error")
        return
        
    # Allow providers to have explicit simple streaming handlers if configured, 
    # otherwise fallback to generic stream wrapper with injected options
    if hasattr(provider, "stream_simple") and provider.stream_simple != provider.stream:
        async for event in _with_error_handling(provider.stream_simple(model, context, options), model):
            yield event
    else:
        async for event in _with_error_handling(provider.stream(model, context, options), model):
            yield event


async def complete_simple(
    model: ModelDef,
    context: Context,
    options: Optional[StreamOptions] = None,
) -> AssistantMessage:
    """
    Non-streaming helper for unified reasoning. Returns final AssistantMessage.
    """
    from .types import AssistantMessage, TextContent, ThinkingContent, ToolUseContent, TokenUsage, StopEvent, ErrorEvent

    blocks = []
    usage = TokenUsage()
    stop_reason = "stop"
    error_message = None

    async for event in stream_simple(model, context, options):
        if event.type == "text_end":
            blocks.append(TextContent(text=event.text))
        elif event.type == "thinking_end":
            blocks.append(ThinkingContent(thinking=event.thinking))
        elif event.type == "tool_use_end":
            blocks.append(event.tool_use)
        elif event.type == "stop":
            usage = event.usage
            stop_reason = event.stop_reason
        elif event.type == "error":
            usage = event.error.usage
            blocks = event.error.content
            stop_reason = event.reason
            error_message = event.error.error_message
            break

    return AssistantMessage(
        role="assistant",
        content=blocks,
        stop_reason=stop_reason,
        usage=usage,
        provider=model.provider,
        model=model.id,
        api=model.api,
        error_message=error_message
    )
