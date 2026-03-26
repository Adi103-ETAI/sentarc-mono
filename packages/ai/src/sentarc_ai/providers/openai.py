"""OpenAI provider — supports both /completions and /responses endpoints."""
from __future__ import annotations
import json
from typing import AsyncIterator, Optional, Any

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk

from ..types import (
    Context, ModelDef, StreamEvent, Role,
    TokenUsage, ReasoningEffort,
    StartEvent, TextStartEvent, TextDeltaEvent, TextEndEvent,
    ThinkingStartEvent, ThinkingDeltaEvent, ThinkingEndEvent,
    ToolUseStartEvent, ToolUseDeltaEvent, ToolUseEndEvent,
    StopEvent, ErrorEvent, ToolUseContent, ToolCallContent,
)
from ..env import get_env_api_key
# ... (imports)

from ..transform_messages import to_openai_messages, transform_messages


class OpenAIProvider:
    def __init__(self):
        self.api = "openai"

    async def stream(
        self,
        model: "ModelDef",
        context: "Context",
        options: Optional[Any] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream from OpenAI API."""
        
        reasoning = options.reasoning_effort if options else ReasoningEffort.NONE
        
        resolved_api_key = getattr(options, "api_key", None) if options else None
        api_key = resolved_api_key or get_env_api_key(model.provider)
        if not api_key:
            raise RuntimeError(f"No API key found for {model.provider}")

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=model.base_url,
            default_headers=model.extra_headers
        )

        # Convert messages
        transformed = transform_messages(context.messages, model)
        messages = to_openai_messages(transformed)
        
        if context.system_prompt:
            role = "developer" if (model.supports_thinking or model.id.startswith("o")) else "system"
            messages.insert(0, {"role": role, "content": context.system_prompt})

        params = {
            "model": model.id,
            "messages": messages,
            "stream": True,
        }

        if context.tools:
            params["tools"] = [{
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters
                }
            } for t in context.tools]
            tool_choice = getattr(context, "tool_choice", None)
            if tool_choice:
                params["tool_choice"] = tool_choice

        if model.supports_thinking and reasoning != ReasoningEffort.NONE:
             params["reasoning_effort"] = reasoning.value

        yield StartEvent(model=model.id)
        
        try:
            stream_resp = await client.chat.completions.create(**params)
            
            text_started = False
            current_tool_call = None
            
            async for chunk in stream_resp:
                if not chunk.choices: continue
                
                delta = chunk.choices[0].delta
                
                # Content / Text
                if delta.content:
                    if not text_started:
                        yield TextStartEvent()
                        text_started = True
                    yield TextDeltaEvent(text=delta.content)
                
                # Reasoning Content
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    yield ThinkingDeltaEvent(thinking=delta.reasoning_content)
                    
                # Tool Calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.id:
                            current_tool_call = {
                                "index": tc.index, 
                                "id": tc.id, 
                                "name": tc.function.name, 
                                "args": ""
                            }
                            yield ToolUseStartEvent(id=tc.id, name=tc.function.name)
                        
                        if tc.function and tc.function.arguments:
                            if current_tool_call and current_tool_call["index"] == tc.index:
                                current_tool_call["args"] += tc.function.arguments
                                yield ToolUseDeltaEvent(id=current_tool_call["id"], partial_input=tc.function.arguments)

                if chunk.usage:
                    u = chunk.usage
                    usage = TokenUsage(
                        input_tokens=u.prompt_tokens,
                        output_tokens=u.completion_tokens,
                    )
                    yield StopEvent(stop_reason="end_turn", usage=usage)
                    return

            if current_tool_call:
                try:
                    args = json.loads(current_tool_call["args"])
                    yield ToolUseEndEvent(tool_use=ToolCallContent( # Using ToolCallContent from types
                         id=current_tool_call["id"],
                         name=current_tool_call["name"],
                         arguments=args
                    ))
                except json.JSONDecodeError:
                    raise RuntimeError("Failed to parse tool arguments")
            
            if text_started:
                yield TextEndEvent(text="")
                
            yield StopEvent(stop_reason="end_turn", usage=TokenUsage())
            
        except Exception as e:
            raise RuntimeError(str(e))



