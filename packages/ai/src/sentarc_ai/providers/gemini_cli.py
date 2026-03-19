"""
Google Cloud Code Assist API Provider.
"""
from __future__ import annotations
import json
import httpx
import asyncio
from typing import AsyncIterator, Optional, Any, Dict, List

from ..types import (
    Context, ModelDef, StreamEvent, ErrorEvent, Role,
    TokenUsage, StartEvent, TextStartEvent, TextDeltaEvent, TextEndEvent,
    ThinkingStartEvent, ThinkingDeltaEvent, ThinkingEndEvent,
    ToolUseStartEvent, ToolUseDeltaEvent, ToolUseEndEvent,
    StopEvent, ToolCallContent, ImageContent, TextContent
)
from ..transform_messages import transform_messages

DEFAULT_ENDPOINT = "https://cloudcode-pa.googleapis.com"
ANTIGRAVITY_DAILY_ENDPOINT = "https://daily-cloudcode-pa.sandbox.googleapis.com"
ANTIGRAVITY_ENDPOINT_FALLBACKS = [ANTIGRAVITY_DAILY_ENDPOINT, DEFAULT_ENDPOINT]

GEMINI_CLI_HEADERS = {
    "User-Agent": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "X-Goog-Api-Client": "gl-node/22.17.0",
    "Client-Metadata": json.dumps({
        "ideType": "IDE_UNSPECIFIED",
        "platform": "PLATFORM_UNSPECIFIED",
        "pluginType": "GEMINI",
    }),
}

class GeminiCliProvider:
    def __init__(self):
        self.api = "google-gemini-cli"

    async def stream(
        self,
        model: "ModelDef",
        context: "Context",
        options: Optional[Any] = None, 
    ) -> AsyncIterator[StreamEvent]:
        """Stream from Cloud Code Assist API."""
        
        auth_data = getattr(options, "api_key", None)
        if not auth_data:
             raise RuntimeError("Google Cloud Code Assist requires OAuth authentication (passed via apiKey).")

        try:
             # Expecting JSON string with token and projectId
             creds = json.loads(auth_data)
             access_token = creds.get("token")
             project_id = creds.get("projectId")
        except:
             raise RuntimeError("Invalid Google Cloud Code Assist credentials format.")

        if not access_token or not project_id:
             raise RuntimeError("Missing token or projectId in Google Cloud credentials.")

        is_antigravity = model.provider == "google-antigravity"
        base_url = model.base_url.strip() if model.base_url else None
        endpoints = [base_url] if base_url else (ANTIGRAVITY_ENDPOINT_FALLBACKS if is_antigravity else [DEFAULT_ENDPOINT])
        
        request_body = self._build_request(model, context, project_id, options)
        
        headers = GEMINI_CLI_HEADERS.copy()
        headers["Authorization"] = f"Bearer {access_token}"
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"
        
        if model.extra_headers:
             headers.update(model.extra_headers)

        # Retry logic and endpoint fallback
        last_error = None
        for endpoint in endpoints:
             url = f"{endpoint}/v1internal:streamGenerateContent?alt=sse"
             try:
                 async for event in self._stream_from_url(url, request_body, headers, model):
                      yield event
                 return # Success
             except Exception as e:
                 last_error = e
                 continue
                 
        raise RuntimeError(f"All endpoints failed. Last error: {str(last_error)}")

    async def _stream_from_url(self, url: str, body: Dict, headers: Dict, model: ModelDef) -> AsyncIterator[StreamEvent]:
         async with httpx.AsyncClient(timeout=120.0) as client:
              async with client.stream("POST", url, json=body, headers=headers) as response:
                    if response.status_code != 200:
                         err_text = await response.aread()
                         raise Exception(f"API Error {response.status_code}: {err_text.decode('utf-8')}")
                    
                    yield StartEvent(model=model.id)
                    
                    current_block_type = None # "thinking", "text", "toolCall"
                    current_text = ""
                    current_thinking = ""
                    current_tool_call = None
                    
                    async for line in response.aiter_lines():
                         if not line.startswith("data:"): continue
                         json_str = line[5:].strip()
                         if not json_str: continue
                         
                         try:
                              chunk = json.loads(json_str)
                              # Parse chunk structure relative to google-gemini-cli.ts
                              # chunk.response.candidates[0].content.parts[...]
                              
                              response_data = chunk.get("response", {})
                              candidates = response_data.get("candidates", [])
                              if not candidates: continue
                              candidate = candidates[0]
                              
                              content = candidate.get("content", {})
                              parts = content.get("parts", [])
                              
                              for part in parts:
                                   # Text / Thought
                                   if "text" in part:
                                        text = part["text"]
                                        is_thought = part.get("thought", False) 
                                        
                                        # State transition
                                        if is_thought:
                                             if current_block_type != "thinking":
                                                  if current_block_type == "text":
                                                       yield TextEndEvent(text=current_text)
                                                  yield ThinkingStartEvent()
                                                  current_block_type = "thinking"
                                                  current_thinking = ""
                                             
                                             current_thinking += text
                                             yield ThinkingDeltaEvent(thinking=text)
                                             
                                        else:
                                             if current_block_type != "text":
                                                  if current_block_type == "thinking":
                                                       yield ThinkingEndEvent(thinking=current_thinking)
                                                  yield TextStartEvent()
                                                  current_block_type = "text"
                                                  current_text = ""
                                                  
                                             current_text += text
                                             yield TextDeltaEvent(text=text)

                                   # Tool Call
                                   elif "functionCall" in part:
                                        fc = part["functionCall"]
                                        if current_block_type:
                                             if current_block_type == "text": yield TextEndEvent(text=current_text)
                                             elif current_block_type == "thinking": yield ThinkingEndEvent(thinking=current_thinking)
                                             current_block_type = None
                                        
                                        # Gemini CLI might not stream tool calls with deltas like OpenAI?
                                        # TS implementation handles full function call in one part usually for Gemini?
                                        # Or creates a new block.
                                        
                                        call_id = fc.get("id") or f"{fc.get('name')}_{asyncio.get_event_loop().time()}" # Fallback ID
                                        yield ToolUseStartEvent(id=call_id, name=fc.get("name"))
                                        yield ToolUseDeltaEvent(id=call_id, partial_input=json.dumps(fc.get("args", {})))
                                        yield ToolUseEndEvent(tool_use=ToolUseContent(
                                             id=call_id,
                                             name=fc.get("name"),
                                             arguments=fc.get("args", {})
                                        ))
                              
                              usage_meta = response_data.get("usageMetadata")
                              if usage_meta:
                                   usage = TokenUsage(
                                        input_tokens=usage_meta.get("promptTokenCount", 0),
                                        output_tokens=usage_meta.get("candidatesTokenCount", 0),
                                        total_tokens=usage_meta.get("totalTokenCount", 0)
                                   )
                                   # If finishReason is present, we might want to yield StopEvent?
                                   # But loop continues.
                                   pass
                                   
                         except Exception:
                              continue

                    # Close open blocks
                    if current_block_type == "text":
                         yield TextEndEvent(text=current_text)
                    elif current_block_type == "thinking":
                         yield ThinkingEndEvent(thinking=current_thinking)
                         
                    yield StopEvent(stop_reason="end_turn", usage=TokenUsage())
 
    def _build_request(self, model: ModelDef, context: Context, project_id: str, options: Any) -> Dict:
         # Simplified request builder
         contents = []
         
         msgs = transform_messages(context.messages, model)
         
         if context.system_prompt:
              # Gemini often supports systemInstruction field, but here we transform messages.
              # transform_messages might have handled it or we pass it separately?
              # TS code sends "systemInstruction" in body.
              pass

         # ... (Implementation of conversion logic would be verbose, simplifying for this step)
         # Using a simplified conversion for POC
         
         for msg in msgs:
              role = "user" if msg.role == Role.USER else "model"
              parts = []
              if isinstance(msg.content, str):
                   parts.append({"text": msg.content})
              elif isinstance(msg.content, list):
                   for c in msg.content:
                        if isinstance(c, TextContent): parts.append({"text": c.text})
                        # ... other types
              
              if parts:
                   contents.append({"role": role, "parts": parts})

         body = {
             "project": project_id,
             "model": model.id,
             "request": {
                 "contents": contents,
                 "generationConfig": {
                     "temperature": options.temperature if options else 0.0
                 }
             }
         }
         
         if context.system_prompt:
              body["request"]["systemInstruction"] = {
                   "parts": [{"text": context.system_prompt}]
              }
              
         return body
