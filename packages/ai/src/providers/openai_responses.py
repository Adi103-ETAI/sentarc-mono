"""
OpenAI Responses API provider.
"""
from __future__ import annotations
import os
from typing import AsyncIterator, Optional, Any, Dict

from openai import AsyncOpenAI

from ..types import (
    Context, ModelDef, StreamEvent, ErrorEvent, ReasoningEffort
)
from ..env import get_env_api_key
from .openai_utils import convert_responses_messages, convert_responses_tools, process_responses_stream

OPENAI_TOOL_CALL_PROVIDERS = {"openai", "openai-codex", "opencode"}

class OpenAIResponsesProvider:
    def __init__(self):
        self.api = "openai-responses"

    async def stream(
        self,
        model: "ModelDef",
        context: "Context",
        options: Optional[Any] = None, 
    ) -> AsyncIterator[StreamEvent]:
        """Stream from OpenAI Responses API."""
        
        api_key = get_env_api_key(model.provider)
        if not api_key:
            yield ErrorEvent(error=f"No API key found for {model.provider}")
            return

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=model.base_url,
            default_headers=model.extra_headers
        )

        messages = convert_responses_messages(model, context, OPENAI_TOOL_CALL_PROVIDERS)
        
        params: Dict[str, Any] = {
            "model": model.id,
            "input": messages,
            "stream": True,
        }

        # Cache retention logic (simplified from TS)
        # TS: prompt_cache_key = options?.sessionId if cacheRetention != "none"
        # We can just check options for session_id if available or context
        if options and getattr(options, "session_id", None):
             params["prompt_cache_key"] = options.session_id # type: ignore

        if context.tools:
            params["tools"] = convert_responses_tools(context.tools)

        # Reasoning effort
        reasoning = options.getattr("reasoning_effort", None) if options else None
        if reasoning:
             params["reasoning"] = {"effort": reasoning.value}

        try:
            # client.responses.create might not exist in all SDK versions yet, 
            # but we assume it does based on the TS port requirement.
            if not hasattr(client, "responses"):
                 yield ErrorEvent(error="OpenAI Python SDK does not support 'responses' API yet.")
                 return

            stream_resp = await client.responses.create(**params)
            
            async for event in process_responses_stream(stream_resp, model):
                yield event

        except Exception as e:
            yield ErrorEvent(error=str(e))
