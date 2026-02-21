"""
Shared utilities for OpenAI Responses and Codex implementations.
"""
from __future__ import annotations
import json
import re
from typing import Any, AsyncIterator, Callable, Optional, Set, List, Dict, Union

from ..types import (
    Context, ModelDef, StreamEvent, Role, TokenUsage,
    StartEvent, TextStartEvent, TextDeltaEvent, TextEndEvent,
    ThinkingStartEvent, ThinkingDeltaEvent, ThinkingEndEvent,
    ToolUseStartEvent, ToolUseDeltaEvent, ToolUseEndEvent,
    StopEvent, ErrorEvent, ToolUseContent, ToolResultContent,
    AssistantMessage, ToolResultMessage, Message, ContentBlock,
    TextContent, ThinkingContent, ToolCallContent, ImageContent
)
from ..transform_messages import transform_messages

def short_hash(s: str) -> str:
    """Fast deterministic hash to shorten long strings."""
    h1 = 0xdeadbeef
    h2 = 0x41c6ce57
    for char in s:
        ch = ord(char)
        h1 = (h1 ^ ch) * 2654435761 & 0xffffffff
        h2 = (h2 ^ ch) * 1597334677 & 0xffffffff
    
    h1 = ((h1 ^ (h1 >> 16)) * 2246822507) ^ ((h2 ^ (h2 >> 13)) * 3266489909) & 0xffffffff
    h2 = ((h2 ^ (h2 >> 16)) * 2246822507) ^ ((h1 ^ (h1 >> 13)) * 3266489909) & 0xffffffff
    
    # Simple base36-like conversion for brevity in Python
    def to_base36(num):
        chars = "0123456789abcdefghijklmnopqrstuvwxyz"
        if num == 0: return "0"
        res = ""
        while num > 0:
            res = chars[num % 36] + res
            num //= 36
        return res

    return to_base36(h2) + to_base36(h1)

def convert_responses_messages(
    model: ModelDef,
    context: Context,
    allowed_tool_call_providers: Set[str],
    include_system_prompt: bool = True
) -> List[Dict[str, Any]]:
    """Convert context messages to OpenAI Responses API format."""
    messages = []
    
    def normalize_tool_call_id(id_val: str, _model: ModelDef, _msg: AssistantMessage) -> str:
        if model.provider not in allowed_tool_call_providers:
            return id_val
        if "|" not in id_val:
            return id_val
        
        call_id, item_id = id_val.split("|", 1)
        sanitized_call_id = re.sub(r"[^a-zA-Z0-9_-]", "_", call_id)
        sanitized_item_id = re.sub(r"[^a-zA-Z0-9_-]", "_", item_id)
        
        if not sanitized_item_id.startswith("fc"):
            sanitized_item_id = f"fc_{sanitized_item_id}"
            
        norm_call_id = sanitized_call_id[:64].rstrip("_")
        norm_item_id = sanitized_item_id[:64].rstrip("_")
        
        return f"{norm_call_id}|{norm_item_id}"

    transformed = transform_messages(
        context.messages, 
        model, 
        normalize_tool_call_id=normalize_tool_call_id
    )

    if include_system_prompt and context.system_prompt:
        # OpenAI "developer" role for reasoning models, "system" otherwise
        # In this specific context (Responses API), "developer" seems standard for newer models?
        # The TS code uses model.reasoning check.
        # Assuming model.supports_thinking or similar flag implies reasoning capability.
        # Check types.py for ModelDef properties. 
        # Using "developer" if supports_thinking, else "system" as a heuristic.
        role = "developer" if model.supports_thinking else "system"
        messages.append({
            "role": role,
            "content": context.system_prompt
        })

    msg_index = 0
    for msg in transformed:
        if msg.role == Role.USER:
            if isinstance(msg.content, str):
                messages.append({
                    "role": "user",
                    "content": [{"type": "input_text", "text": msg.content}]
                })
            else:
                content_parts = []
                for item in msg.content:
                    if isinstance(item, TextContent):
                        content_parts.append({
                            "type": "input_text",
                            "text": item.text
                        })
                    elif isinstance(item, ImageContent):
                         # Assuming ImageContent has mime_type and data (base64)
                         # types.py ImageContent definition needs verification, 
                         # usually it has data and media_type (or mime_type)
                         # Only include if model supports images (checked via model.input_modalities usually)
                         # TS code: !model.input.includes("image")
                         # Python: model.input_modalities (list of strings)
                         if "image" in (model.input_modalities or []):
                            content_parts.append({
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": f"data:{item.media_type};base64,{item.data}"
                            })
                
                if content_parts:
                    messages.append({
                        "role": "user",
                        "content": content_parts
                    })
                    
        elif msg.role == Role.ASSISTANT:
            output_parts = []
            assistant_msg: AssistantMessage = msg
            
            # Check for different model provenance
            is_different_model = (
                assistant_msg.model != model.id and 
                assistant_msg.provider == model.provider and
                assistant_msg.api == model.api
            )
            
            # Message content in Python is list[ContentBlock]
            # ContentBlock = TextContent | ThinkingContent | ToolCallContent | ...
            
            if isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ThinkingContent):
                        if block.thinking_signature:
                             try:
                                 reasoning_item = json.loads(block.thinking_signature)
                                 output_parts.append(reasoning_item)
                             except:
                                 pass
                    elif isinstance(block, TextContent):
                        # OpenAI Responses API uses 'message' items for text output
                        msg_id = getattr(block, 'text_signature', None)
                        if not msg_id:
                            msg_id = f"msg_{msg_index}"
                        elif len(msg_id) > 64:
                            msg_id = f"msg_{short_hash(msg_id)}"
                        
                        output_parts.append({
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": block.text}],
                            "status": "completed",
                            "id": msg_id
                        })
                    elif isinstance(block, ToolCallContent):
                        call_id_raw = block.id.split("|")
                        call_id = call_id_raw[0]
                        item_id = call_id_raw[1] if len(call_id_raw) > 1 else None
                        
                        # Strip itemId if from different model to avoid validation errors
                        if is_different_model and item_id and item_id.startswith("fc_"):
                            item_id = None
                            
                        output_parts.append({
                            "type": "function_call",
                            "id": item_id,
                            "call_id": call_id,
                            "name": block.name,
                            "arguments": json.dumps(block.arguments)
                        })
            
            if output_parts:
                messages.extend(output_parts)
                
        elif msg.role == Role.TOOL_RESULT:
             # ToolResultMessage
             msg: ToolResultMessage
             
             text_parts = [c.text for c in msg.content if isinstance(c, TextContent)]
             text_result = "\n".join(text_parts)
             has_images = any(isinstance(c, ImageContent) for c in msg.content)
             
             has_text = len(text_result) > 0
             call_id_parts = msg.tool_call_id.split("|")
             call_id = call_id_parts[0]
             
             messages.append({
                 "type": "function_call_output",
                 "call_id": call_id,
                 "output": text_result if has_text else "(see attached image)"
             })
             
             if has_images and "image" in (model.input_modalities or []):
                 content_parts = [
                     {"type": "input_text", "text": "Attached image(s) from tool result:"}
                 ]
                 for block in msg.content:
                     if isinstance(block, ImageContent):
                         content_parts.append({
                             "type": "input_image",
                             "detail": "auto",
                             "image_url": f"data:{block.media_type};base64,{block.data}"
                         })
                 messages.append({
                     "role": "user",
                     "content": content_parts
                 })
                 
        msg_index += 1
        
    return messages

def convert_responses_tools(tools: List[Any], options: Optional[Dict] = None) -> List[Dict]:
    """Convert tools to OpenAI format."""
    strict = options.get("strict", False) if options else False
    return [{
        "type": "function",
        "name": t.name,
        "description": t.description,
        "parameters": t.parameters,
        "strict": strict
    } for t in tools]


async def process_responses_stream(
    openai_stream: AsyncIterator[Any],
    model: ModelDef
) -> AsyncIterator[StreamEvent]:
    """Process OpenAI Responses API stream."""
    
    current_item_type = None # "reasoning", "message", "function_call"
    current_block_type = None # "thinking", "text", "toolCall"
    current_text = ""
    current_thinking = ""
    current_tool_call: Optional[Dict] = None
    
    # Store summaries for thinking blocks
    current_thinking_summary = [] 
    
    yield StartEvent(model=model.id)
    
    try:
        async for event in openai_stream:
            # Event structure in python SDK might differ slightly from TS raw SSE
            # Assuming openai_stream yields objects with 'type' attribute 
            # Or dictionaries if we use raw response parsing (which we might need for Responses API beta)
            
            # The python SDK for Responses API is experimental. 
            # If using `client.responses.create(stream=True)`, it yields events.
            
            # Let's assume event is a dictionary or object with attributes.
            # Using dictionary access for safety if it's raw JSON.
            
            e_type = getattr(event, "type", None) or event.get("type")
            
            if e_type == "response.output_item.added":
                item = getattr(event, "item", None) or event.get("item", {})
                i_type = item.get("type")
                
                if i_type == "reasoning":
                    current_item_type = "reasoning"
                    current_block_type = "thinking"
                    current_thinking = ""
                    current_thinking_summary = []
                    yield ThinkingStartEvent()
                    
                elif i_type == "message":
                    current_item_type = "message"
                    current_block_type = "text"
                    current_text = ""
                    yield TextStartEvent()
                    
                elif i_type == "function_call":
                    current_item_type = "function_call"
                    current_block_type = "toolCall"
                    current_tool_call = {
                        "id": f"{item.get('call_id')}|{item.get('id')}",
                        "name": item.get('name'),
                        "args": "",
                        "partial_json": item.get('arguments', "")
                    }
                    yield ToolUseStartEvent(id=current_tool_call["id"], name=current_tool_call["name"])
            
            elif e_type == "response.reasoning_summary_part.added":
                # Only tracking internal state, no yield
                if current_item_type == "reasoning":
                    part = getattr(event, "part", None) or event.get("part")
                    current_thinking_summary.append(part)

            elif e_type == "response.reasoning_summary_text.delta":
                 if current_item_type == "reasoning" and current_block_type == "thinking":
                     delta = getattr(event, "delta", "") or event.get("delta", "")
                     if delta:
                         current_thinking += delta
                         # Also update last summary part text internally? 
                         yield ThinkingDeltaEvent(thinking=delta)
                         
            elif e_type == "response.reasoning_summary_part.done":
                 if current_item_type == "reasoning":
                     # Add double newline separator
                     yield ThinkingDeltaEvent(thinking="\n\n")

            elif e_type == "response.content_part.added":
                 # message content part added (text or refusal)
                 pass

            elif e_type == "response.output_text.delta":
                 if current_item_type == "message" and current_block_type == "text":
                     delta = getattr(event, "delta", "") or event.get("delta", "")
                     if delta:
                         current_text += delta
                         yield TextDeltaEvent(text=delta)

            elif e_type == "response.refusal.delta":
                 if current_item_type == "message" and current_block_type == "text":
                     delta = getattr(event, "delta", "") or event.get("delta", "")
                     if delta:
                         current_text += delta
                         yield TextDeltaEvent(text=delta)

            elif e_type == "response.function_call_arguments.delta":
                 if current_item_type == "function_call" and current_block_type == "toolCall":
                     delta = getattr(event, "delta", "") or event.get("delta", "")
                     if delta:
                         current_tool_call["partial_json"] += delta
                         yield ToolUseDeltaEvent(id=current_tool_call["id"], partial_input=delta)

            elif e_type == "response.function_call_arguments.done":
                 if current_item_type == "function_call":
                     args = getattr(event, "arguments", "") or event.get("arguments", "")
                     current_tool_call["partial_json"] = args

            elif e_type == "response.output_item.done":
                 item = getattr(event, "item", None) or event.get("item", {})
                 i_type = item.get("type")
                 
                 if i_type == "reasoning" and current_block_type == "thinking":
                     # Finalize thinking
                     yield ThinkingEndEvent(thinking=current_thinking)
                     current_block_type = None
                     
                 elif i_type == "message" and current_block_type == "text":
                     yield TextEndEvent(text=current_text)
                     current_block_type = None
                     
                 elif i_type == "function_call":
                     # Parse args
                     args_str = current_tool_call.get("partial_json", "{}")
                     try:
                         args = json.loads(args_str)
                     except:
                         args = {}
                     
                     yield ToolUseEndEvent(tool_use=ToolUseContent(
                         id=current_tool_call["id"],
                         name=current_tool_call["name"],
                         arguments=args
                     ))
                     current_block_type = None

            elif e_type == "response.completed":
                 # Handle usage and finish
                 resp = getattr(event, "response", None) or event.get("response", {})
                 usage_data = resp.get("usage", {})
                 if usage_data:
                     usage = TokenUsage(
                         input_tokens=usage_data.get("input_tokens", 0),
                         output_tokens=usage_data.get("output_tokens", 0),
                         total_tokens=usage_data.get("total_tokens", 0)
                     )
                     # Check status for stop reason
                     status = resp.get("status")
                     stop_reason = "end_turn"
                     if status == "incomplete": stop_reason = "max_tokens"
                     elif status == "failed": stop_reason = "error" # Should have been caught earlier?
                     
                     yield StopEvent(stop_reason=stop_reason, usage=usage)
                     return

            elif e_type == "error":
                 # Stream error
                 code = getattr(event, "code", "") or event.get("code")
                 msg = getattr(event, "message", "") or event.get("message")
                 raise RuntimeError(f"Error {code}: {msg}")

    except Exception as e:
        raise RuntimeError(str(e))
