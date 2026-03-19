"""
Message transformation — packages/ai/src/transform-messages.ts from sentarc-mono.
"""
from __future__ import annotations
import json
import time
from typing import Optional, Callable, Any
from dataclasses import replace
from .types import (
    Message, Role, TextContent, ImageContent, ToolCallContent, 
    AssistantMessage, ToolResultMessage, ModelDef, ContentBlock,
    TokenUsage, ThinkingContent, ToolUseContent
)
from .utils.sanitize_unicode import sanitize_surrogates

def transform_messages(
    messages: list[Message],
    model: ModelDef,
    normalize_tool_call_id: Optional[Callable[[str, ModelDef, AssistantMessage], str]] = None,
) -> list[Message]:
    """
    Transform messages for the target model.
    Handles:
    - Thinking blocks (strip/keep based on model support)
    - Tool call ID normalization (cross-provider)
    - Synthetic tool results for orphaned tool calls
    - filtering out error/aborted messages
    """
    
    # Map original tool call IDs to normalized IDs
    tool_call_id_map: dict[str, str] = {}
    
    # -------------------------------------------------------
    # Pass 1: Transform content (thinking, tool IDs)
    # -------------------------------------------------------
    transformed: list[Message] = []
    
    for msg in messages:
        # Sanitize system prompts or generic string content
        if msg.role in (Role.USER, Role.SYSTEM) and isinstance(msg.content, str):
            transformed.append(replace(msg, content=sanitize_surrogates(msg.content)))
            continue
            
        # User messages with blocks
        if msg.role == Role.USER:
            if isinstance(msg.content, list):
                new_blocks = []
                for block in msg.content:
                    if isinstance(block, TextContent):
                        new_blocks.append(replace(block, text=sanitize_surrogates(block.text)))
                    else:
                        new_blocks.append(block)
                transformed.append(replace(msg, content=new_blocks))
            else:
                 transformed.append(msg)
            continue
            
        # Tool results - check if ID needs remapping
        if msg.role == Role.TOOL:
            # Cast to ToolResultMessage if possible, else treat as generic Message
            tool_id = msg.tool_call_id or "" 
            
            normalized_id = tool_call_id_map.get(tool_id)
            if normalized_id and normalized_id != tool_id:
                # Create a copy with new ID using ToolResultMessage if applicable or generic
                new_msg = ToolResultMessage(
                    role=Role.TOOL,
                    content=msg.content,
                    tool_call_id=normalized_id,
                    is_error=getattr(msg, "is_error", False),
                    timestamp=getattr(msg, "timestamp", None)
                )
                transformed.append(new_msg)
            else:
                transformed.append(msg)
            continue
            
        # Assistant messages
        if msg.role == Role.ASSISTANT:
            # Check stop reason - skip if error/aborted
            stop_reason = getattr(msg, "stop_reason", None)
            if stop_reason in ["error", "aborted"]:
                continue
                
            # Determine if this message is from the same model
            msg_provider = getattr(msg, "provider", None)
            msg_model = getattr(msg, "model", None)
            
            # If provider/model not set (generic Message), assume same model or unknown
            # If unknown, we usually treat as same unless restricted?
            # TypeScript logic: isSameModel = (msg.provider === model.provider && ...)
            # If msg is generic, these are None. 
            # We'll treat None as "different" or "generic"? 
            # Let's assume generic messages (from user input history) might need processing.
            
            is_same_model = (
                msg_provider == model.provider and
                msg_model == model.id
            )
            
            # Transform content blocks
            new_content: list[ContentBlock] = []
            
            # Normalize content to list of blocks
            blocks = []
            if isinstance(msg.content, str):
                blocks = [TextContent(text=msg.content)]
            elif isinstance(msg.content, list):
                blocks = msg.content
            
            for block in blocks:
                if isinstance(block, ThinkingContent):
                    # Sanitize thinking block strings
                    sanitized_thinking = sanitize_surrogates(block.thinking)
                    # Same model: keep thinking (with signature)
                    if is_same_model and block.signature:
                        new_content.append(replace(block, thinking=sanitized_thinking))
                        continue
                        
                    if not sanitized_thinking or not sanitized_thinking.strip():
                         if is_same_model and block.signature: # Encrypted reasoning
                             new_content.append(replace(block, thinking=sanitized_thinking))
                         continue
                        
                    if is_same_model:
                        new_content.append(replace(block, thinking=sanitized_thinking))
                    else:
                        # Convert to text if different model
                        new_content.append(TextContent(text=sanitized_thinking))
                        
                elif isinstance(block, TextContent):
                    new_content.append(replace(block, text=sanitize_surrogates(block.text)))
                    
                elif isinstance(block, (ToolCallContent, ToolUseContent)):
                    # Normalize tool call
                    tool_id = block.id
                    args = block.arguments
                    name = block.name
                    
                    final_id = tool_id
                    
                    if not is_same_model and normalize_tool_call_id:
                        # Create a temp AssistantMessage wrapper for context if needed
                        # We pass the original msg (or cast it)
                        normalized = normalize_tool_call_id(tool_id, model, msg) # type: ignore
                        if normalized != tool_id:
                            tool_call_id_map[tool_id] = normalized
                            final_id = normalized
                            
                    # Reconstruct block
                    if isinstance(block, ToolCallContent):
                        new_content.append(ToolCallContent(id=final_id, name=name, arguments=args))
                    else:
                         new_content.append(ToolUseContent(id=final_id, name=name, arguments=args))
                         
                else:
                    new_content.append(block)
            
            # Update tool_calls list on the message for convenience
            new_tool_calls = [b for b in new_content if isinstance(b, (ToolCallContent, ToolUseContent))]
            
            # Create new assistant message
            new_msg = AssistantMessage(
                role=Role.ASSISTANT,
                content=new_content,
                tool_calls=new_tool_calls, # type: ignore
                stop_reason=stop_reason,
                usage=getattr(msg, "usage", None) or TokenUsage(),
                provider=msg_provider,
                model=msg_model
            )
            transformed.append(new_msg)
            continue
            
        # Fallback for SYSTEM or other containing str
        if isinstance(msg.content, str):
            transformed.append(replace(msg, content=sanitize_surrogates(msg.content)))
        else:
            transformed.append(msg)
        
        
    # -------------------------------------------------------
    # Pass 2: Insert synthetic tool results for orphans
    # -------------------------------------------------------
    final_messages: list[Message] = []
    pending_tool_calls: list[ToolCallContent | ToolUseContent] = []
    existing_tool_result_ids: set[str] = set()
    
    for msg in transformed:
        if msg.role == Role.ASSISTANT:
            # If we have pending orphans, close them out
            if pending_tool_calls:
                for tc in pending_tool_calls:
                    if tc.id not in existing_tool_result_ids:
                        final_messages.append(ToolResultMessage(
                            role=Role.TOOL,
                            tool_call_id=tc.id,
                            content="No result provided",
                            is_error=True,
                            timestamp=time.time()
                        ))
                pending_tool_calls = []
                existing_tool_result_ids = set()
            
            # Track new tool calls
            # msg.tool_calls is populated on AssistantMessage
            if msg.tool_calls:
                 pending_tool_calls = msg.tool_calls
                 existing_tool_result_ids = set()
            
            final_messages.append(msg)
            
        elif msg.role == Role.TOOL:
            if msg.tool_call_id:
                existing_tool_result_ids.add(msg.tool_call_id)
            final_messages.append(msg)
            
        elif msg.role == Role.USER:
            # User interrupts tool flow
            if pending_tool_calls:
                for tc in pending_tool_calls:
                     if tc.id not in existing_tool_result_ids:
                        final_messages.append(ToolResultMessage(
                            role=Role.TOOL,
                            tool_call_id=tc.id,
                            content="No result provided",
                            is_error=True,
                            timestamp=time.time()
                        ))
                pending_tool_calls = []
                existing_tool_result_ids = set()
            final_messages.append(msg)
            
        else:
            final_messages.append(msg)
            
    # Final flush
    if pending_tool_calls:
        for tc in pending_tool_calls:
             if tc.id not in existing_tool_result_ids:
                final_messages.append(ToolResultMessage(
                    role=Role.TOOL,
                    tool_call_id=tc.id,
                    content="No result provided",
                    is_error=True,
                    timestamp=time.time()
                ))
                
    return final_messages


# -------------------------------------------------------
# Provider-Specific Formatters (Adapters)
# -------------------------------------------------------

def to_anthropic_messages(messages: list[Message]) -> list[dict]:
    result = []
    for m in messages:
        if m.role == Role.SYSTEM: continue
        
        if m.role == Role.TOOL:
            content = _anthropic_content(m.content)
            # Ensure it's a list for tool_result
            final_content = []
            if isinstance(m.content, list):
                 for b in m.content:
                     if isinstance(b, ToolResultContent):
                         final_content.append({
                             "type": "tool_result",
                             "tool_use_id": b.tool_call_id,
                             "content": b.content, 
                             "is_error": getattr(m, "is_error", False)
                         })
            else:
                 final_content.append({
                     "type": "tool_result",
                     "tool_use_id": m.tool_call_id or "unknown",
                     "content": str(m.content),
                     "is_error": getattr(m, "is_error", False)
                 })
                 
            result.append({"role": "user", "content": final_content})

        elif m.role == Role.ASSISTANT:
            content = []
            if isinstance(m.content, str) and m.content:
                content.append({"type": "text", "text": m.content})
            elif isinstance(m.content, list):
                for b in m.content:
                    if isinstance(b, TextContent):
                        content.append({"type": "text", "text": b.text})
                    elif isinstance(b, ThinkingContent):
                        content.append({
                            "type": "thinking",
                            "thinking": b.thinking,
                            "signature": b.signature
                        })
                    elif isinstance(b, (ToolCallContent, ToolUseContent)):
                        content.append({
                            "type": "tool_use",
                            "id": b.id,
                            "name": b.name,
                            "input": b.arguments
                        })
            
            result.append({"role": "assistant", "content": content})
            
        else:
            # User
            result.append({"role": "user", "content": _anthropic_content(m.content)})
            
    return result

def _anthropic_content(c):
    if isinstance(c, str): return c
    if isinstance(c, list):
        out = []
        for b in c:
            if isinstance(b, TextContent):
                 out.append({"type": "text", "text": b.text})
            elif isinstance(b, ImageContent):
                 out.append({
                     "type": "image",
                     "source": {
                         "type": "base64",
                         "media_type": b.media_type,
                         "data": b.data
                     }
                 })
        return out
    return str(c)


def to_openai_messages(messages: list[Message]) -> list[dict]:
    result = []
    for m in messages:
        if m.role == Role.SYSTEM:
            result.append({"role": "system", "content": _openai_content(m.content)})
            continue
            
        if m.role == Role.TOOL:
            result.append({
                "role": "tool",
                "tool_call_id": m.tool_call_id,
                "content": _openai_content(m.content)
            })
            
        elif m.role == Role.ASSISTANT:
            msg = {"role": "assistant"}
            content = _openai_content(m.content)
            if content:
                msg["content"] = content
                
            if m.tool_calls:
                msg["tool_calls"] = [{
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                    }
                } for tc in m.tool_calls]
            result.append(msg)
            
        else:
            result.append({"role": "user", "content": _openai_content(m.content)})
            
    return result

def _openai_content(c):
    if isinstance(c, str): return c
    if isinstance(c, list):
        # Flatten simple text blocks?
        # OpenAI supports list of content parts
        parts = []
        for b in c:
            if isinstance(b, TextContent):
                parts.append({"type": "text", "text": b.text})
            elif isinstance(b, ImageContent):
                parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{b.media_type};base64,{b.data}"
                    }
                })
        # If single text, return str? No, keep list for robustness if mixed
        if len(parts) == 1 and parts[0]["type"] == "text":
            return parts[0]["text"]
        return parts if parts else ""
    return str(c)

# Aliases
to_anthropic = to_anthropic_messages
to_openai    = to_openai_messages

