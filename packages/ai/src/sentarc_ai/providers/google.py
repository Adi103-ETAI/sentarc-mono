"""
Google Gemini provider.
"""
from __future__ import annotations
import os
import json
import asyncio
from typing import AsyncIterator, Optional, Any

try:
    import google.generativeai as genai
    from google.generativeai.types import (
        GenerateContentResponse,
        ContentDict,
        PartDict,
    )
except ImportError:
    # Allow import without package for typing/linting, but stream will fail
    genai = None

from ..types import (
    Context, ModelDef, StreamEvent, Role,
    TokenUsage, ReasoningEffort,
    StartEvent, TextStartEvent, TextDeltaEvent, TextEndEvent,
    ThinkingStartEvent, ThinkingDeltaEvent, ThinkingEndEvent,
    ToolUseStartEvent, ToolUseDeltaEvent, ToolUseEndEvent,
    StopEvent, ErrorEvent,
)
from ..env import get_env_api_key
from ..transform_messages import transform_messages

# ... (imports remain same)

def _convert_messages(messages: list[Message]) -> list[ContentDict]:
    """Convert sentarc-ai messages to Google ContentDict format."""
    contents = []
    for m in messages:
        role = "user" if m.role == Role.USER else "model"
        parts = []
        
        if m.role == Role.TOOL:
            # Tool result
            role = "function" 
            parts.append(PartDict(function_response={
                "name": m.tool_call_id, 
                "response": {"content": m.content} 
            }))
        
        elif isinstance(m.content, str):
            parts.append(PartDict(text=m.content))
            
        elif isinstance(m.content, list):
            for block in m.content:
                if isinstance(block, str):
                     parts.append(PartDict(text=block))
                elif hasattr(block, 'type'):
                    if block.type == "text":
                        parts.append(PartDict(text=block.text))
                    elif block.type == "tool_use":
                        parts.append(PartDict(function_call={
                            "name": block.name,
                            "args": block.arguments
                        }))
                    elif block.type == "tool_result":
                        pass

        if parts:
            contents.append(ContentDict(role=role, parts=parts))
            
    return contents
# ... (rest of _convert_messages implementation unchanged, just signature comment updated if needed, but logic uses iteration so fine)



def _convert_tools(tools) -> list[Any]:
    """Convert sentarc-ai tools to Google Tool format."""
    # SDK expects tools=[{'function_declarations': [...]}]
    funcs = []
    for t in tools:
        # Convert JSON schema types to Google types if needed, 
        # or pass raw if SDK supports it (latest versions do allow dicts).
        funcs.append({
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters
        })
    return [{"function_declarations": funcs}]



class GoogleProvider:
    def __init__(self):
        self.api = "google"

    async def stream(
        self,
        model: "ModelDef",
        context: "Context",
        options: Optional[StreamOptions] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream from Google Gemini API."""
        if not genai:
            raise RuntimeError("google-generativeai package not found. Install it with pip.")

        api_key = get_env_api_key(model.provider)
        if not api_key:
            raise RuntimeError(f"No API key found for {model.provider}")

        genai.configure(api_key=api_key)
        
        generation_config = {
            "temperature": 0.7,
            "max_output_tokens": model.max_output,
        }
        
        model_tools = _convert_tools(context.tools) if context.tools else None
        
        gemini_model = genai.GenerativeModel(
            model_name=model.id,
            generation_config=generation_config,
            tools=model_tools
        )

        transformed_messages = transform_messages(context.messages, model)
        contents = _convert_messages(transformed_messages)
        
        yield StartEvent(model=model.id)
        
        text_started = False
        
        try:
            response = await gemini_model.generate_content_async(
                contents,
                stream=True,
                generation_config=generation_config
            )
            
            async for chunk in response:
                if not chunk.candidates: continue
                
                parts = chunk.candidates[0].content.parts
                for part in parts:
                    if part.text:
                        if not text_started:
                            yield TextStartEvent()
                            text_started = True
                        yield TextDeltaEvent(text=part.text)
                    
                    if part.function_call:
                        fc = part.function_call
                        import uuid
                        import json
                        tc_id = f"call_{fc.name}_{str(uuid.uuid4())[:8]}"
                        
                        yield ToolUseStartEvent(id=tc_id, name=fc.name)
                        args = dict(fc.args)
                        yield ToolUseDeltaEvent(id=tc_id, partial_input=json.dumps(args))
                        yield ToolUseEndEvent(tool_use=ToolUseContent(id=tc_id, name=fc.name, arguments=args))

            if response.usage_metadata:
               u = response.usage_metadata
               usage = TokenUsage(
                   input_tokens=u.prompt_token_count,
                   output_tokens=u.candidates_token_count,
                   total_tokens=u.total_token_count
               )
               yield StopEvent(stop_reason="end_turn", usage=usage)
            else:
               yield StopEvent(stop_reason="end_turn", usage=TokenUsage())

        except Exception as e:
            raise RuntimeError(str(e))
