"""
OpenAI Codex Responses API provider.
"""
from __future__ import annotations

import json
from typing import AsyncIterator, Optional, Any, Dict

import httpx

from ..types import Context, ModelDef, StreamEvent
from ..env import get_env_api_key
from .openai_utils import (
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)

CODEX_TOOL_CALL_PROVIDERS = {"openai", "openai-codex", "opencode"}
DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"


class OpenAICodexProvider:
    def __init__(self):
        self.api = "openai-codex-responses"

    async def stream(
        self,
        model: "ModelDef",
        context: "Context",
        options: Optional[Any] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream from OpenAI Codex Responses API."""

        resolved_api_key = getattr(options, "api_key", None) if options else None
        api_key = resolved_api_key or get_env_api_key(model.provider)
        if not api_key:
            raise RuntimeError(f"No API key found for {model.provider}")

        messages = convert_responses_messages(
            model,
            context,
            CODEX_TOOL_CALL_PROVIDERS,
            include_system_prompt=False,
        )

        body: Dict[str, Any] = {
            "model": model.id,
            "store": False,
            "stream": True,
            "instructions": context.system_prompt,
            "input": messages,
            "text": {"verbosity": getattr(options, "text_verbosity", "medium") if options else "medium"},
            "include": ["reasoning.encrypted_content"],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }

        if options and getattr(options, "session_id", None):
            body["prompt_cache_key"] = options.session_id

        if context.tools:
            body["tools"] = convert_responses_tools(context.tools, options={"strict": None})

        base_url = model.base_url or DEFAULT_CODEX_BASE_URL
        url = base_url.rstrip("/")
        if url.endswith("/codex/responses"):
            pass
        elif url.endswith("/codex"):
            url += "/responses"
        else:
            url += "/codex/responses"

        account_id = api_key.split("::", 1)[0] if "::" in api_key else None

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if account_id:
            headers["X-OpenAI-Account-ID"] = account_id
        if model.extra_headers:
            headers.update(model.extra_headers)

        try:
            event_generator = self._fetch_events(url, body, headers)
            async for event in process_responses_stream(event_generator, model):
                yield event
        except Exception as e:
            raise RuntimeError(str(e))

    async def _fetch_events(self, url: str, body: Dict[str, Any], headers: Dict[str, str]) -> AsyncIterator[Dict[str, Any]]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=body, headers=headers) as response:
                if response.status_code != 200:
                    err_text = await response.aread()
                    yield {"type": "error", "code": str(response.status_code), "message": err_text.decode("utf-8")}
                    return

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        yield json.loads(data_str)
                    except Exception:
                        continue
