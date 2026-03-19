
import os
import json
import time
import boto3
from typing import AsyncIterator, Optional, Any, Literal
from ..types import (
    ModelDef, Context, AssistantMessage,
    StreamEvent, TextContent, ToolCallContent as ToolCall, ToolResultContent,
    StreamOptions, ReasoningEffort as ThinkingLevel, Message
)
from ..utils.partial_json import parse_streaming_json
from ..transform_messages import transform_messages

def map_thinking_level_to_budget(level: ThinkingLevel, model_id: str) -> int:
    # Default budgets
    defaults = {
        ThinkingLevel.MINIMAL: 1024,
        ThinkingLevel.LOW: 2048,
        ThinkingLevel.MEDIUM: 8192,
        ThinkingLevel.HIGH: 16384,
        ThinkingLevel.XHIGH: 16384, # Clamp xhigh to high for now
    }
    return defaults.get(level, 16384)

class BedrockProvider:
    def __init__(self):
        self.api = "bedrock-converse-stream"

    async def stream(
        self,
        model: ModelDef,
        context: Context,
        options: Optional[StreamOptions] = None,
    ) -> AsyncIterator[StreamEvent]:
        options = options or StreamOptions()
        
        # Initialize boto3 client
        region = os.getenv("AWS_REGION", "us-east-1")
        if "AWS_BEDROCK_REGION" in os.environ:
             region = os.environ["AWS_BEDROCK_REGION"]
             
        client = boto3.client("bedrock-runtime", region_name=region)

        client = boto3.client("bedrock-runtime", region_name=region)

        # Transform & Convert
        transformed = transform_messages(context.messages, model)
        messages = self._convert_messages(transformed)
        
        # Prepare system prompt
        system = []
        if context.system_prompt:
            system.append({"text": context.system_prompt})

        # Inference config
        inference_config = {
            "maxTokens": options.max_tokens or model.max_output or 4096,
            "temperature": options.temperature if options.temperature is not None else 0.7,
        }

        # Tool config
        tool_config = self._convert_tool_config(context.tools)

        # Additional model fields (Thinking)
        additional_model_fields = {}
        if options.thinking_enabled and model.supports_thinking:
             # Check if model supports reasoning/thinking
             # For Claude 3.7 Sonnet / Claude 3.5 Haiku?
             # Implementation matches typescript logic
             if "anthropic.claude" in model.id:
                 budget = options.thinking_budget 
                 if not budget and options.reasoning_effort:
                     budget = map_thinking_level_to_budget(options.reasoning_effort, model.id)
                 
                 if budget:
                    additional_model_fields["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": budget
                    }
                 elif "opus-4-6" in model.id: # Adaptive example
                     pass # handled differently in TS, simplistic here

        args = {
            "modelId": model.id,
            "messages": messages,
            "inferenceConfig": inference_config,
        }
        if system:
            args["system"] = system
        if tool_config:
            args["toolConfig"] = tool_config
        if additional_model_fields:
            args["additionalModelRequestFields"] = additional_model_fields

        # Invoke stream
        response = client.converse_stream(**args)
        stream = response.get("stream")

        output_message = AssistantMessage(
            role="assistant",
            content=[],
            provider=model.provider,
            model=model.id,
        )

        current_block_index = 0
        current_tool_id = ""
        current_tool_name = ""
        current_tool_json = ""
        
        if stream:
            for event in stream:
                if "messageStart" in event:
                    yield StreamEvent(type="start", partial=output_message)
                
                elif "contentBlockStart" in event:
                    start = event["contentBlockStart"]
                    index = start["contentBlockIndex"]
                    if "toolUse" in start["start"]:
                        tool_use = start["start"]["toolUse"]
                        current_tool_id = tool_use["toolUseId"]
                        current_tool_name = tool_use["name"]
                        current_tool_json = ""
                        
                        tool_call = ToolCall(
                            id=current_tool_id,
                            name=current_tool_name,
                            arguments={}
                        )
                        output_message.content.append(tool_call)
                        current_block_index = len(output_message.content) - 1
                        yield StreamEvent(type="toolcall_start", content_index=current_block_index, partial=output_message)
                    elif "text" in start.get("start", {}):
                        # Text block start usually implied or handled in delta
                        pass

                elif "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"]
                    delta_content = delta["delta"]
                    
                    if "text" in delta_content:
                        text = delta_content["text"]
                        
                        # Ensure we have a text block
                        if not output_message.content or not isinstance(output_message.content[-1], TextContent):
                             output_message.content.append(TextContent(text=""))
                             current_block_index = len(output_message.content) - 1
                             yield StreamEvent(type="text_start", content_index=current_block_index, partial=output_message)
                        
                        # Append text
                        block = output_message.content[-1]
                        if isinstance(block, TextContent):
                            block.text += text
                            yield StreamEvent(type="text_delta", content_index=current_block_index, delta=text, partial=output_message)

                    elif "toolUse" in delta_content:
                         input_delta = delta_content["toolUse"]["input"]
                         current_tool_json += input_delta
                         
                         # Update arguments in partial
                         block = output_message.content[current_block_index]
                         if isinstance(block, ToolCall):
                             block.arguments = parse_streaming_json(current_tool_json)
                             yield StreamEvent(type="toolcall_delta", content_index=current_block_index, delta=input_delta, partial=output_message)
                    
                    elif "reasoningContent" in delta_content:
                        # Handle thinking/reasoning
                        # TODO: Implement ThinkingContent support in types.py if not exists
                         pass 

                elif "contentBlockStop" in event:
                    stop = event["contentBlockStop"]
                    index = stop["contentBlockIndex"]
                    block = output_message.content[len(output_message.content) - 1] # Simplified assumption
                    
                    if isinstance(block, ToolCall):
                         block.arguments = parse_streaming_json(current_tool_json)
                         yield StreamEvent(type="toolcall_end", content_index=current_block_index, tool_call=block, partial=output_message)
                    elif isinstance(block, TextContent):
                         yield StreamEvent(type="text_end", content_index=current_block_index, content=block.text, partial=output_message)

                elif "messageStop" in event:
                    stop_reason = event["messageStop"]["stopReason"]
                    reason_map = {
                        "end_turn": "stop",
                        "tool_use": "toolUse",
                        "max_tokens": "length",
                        "stop_sequence": "stop"
                    }
                    output_message.stop_reason = reason_map.get(stop_reason, "stop")

                elif "metadata" in event:
                    usage = event["metadata"]["usage"]
                    output_message.usage.input_tokens = usage["inputTokens"]
                    output_message.usage.output_tokens = usage["outputTokens"]
                    output_message.usage.total_tokens = usage["totalTokens"]
        
        yield StreamEvent(type="done", reason=output_message.stop_reason, message=output_message)


    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        bedrock_messages = []
        for m in messages:
            role = "user" if m.role == "user" else "assistant"
            content = []
            
            if isinstance(m.content, str):
                content.append({"text": m.content})
            elif isinstance(m.content, list):
                for block in m.content:
                    if isinstance(block, TextContent):
                         content.append({"text": block.text})
                    elif isinstance(block, ToolCall):
                         content.append({
                             "toolUse": {
                                 "toolUseId": block.id,
                                 "name": block.name,
                                 "input": block.arguments
                             }
                         })
                    elif isinstance(block, ToolResultContent):
                         # Bedrock expects tool results in a separate structure? 
                         # Actually converse API handles 'user' role with 'toolResult' content blocks
                         pass

            # Handle Tool Results (which are "user" role in Bedrock Converse)
            if m.role == "toolResult":
                 role = "user"
                 content = [{
                     "toolResult": {
                         "toolUseId": m.tool_call_id,
                         "content": [{"text": c.text} for c in m.content if isinstance(c, TextContent)],
                         "status": "error" if m.is_error else "success"
                     }
                 }]
            
            bedrock_messages.append({"role": role, "content": content})
        return bedrock_messages

    def _convert_tool_config(self, tools: Optional[list]) -> Optional[dict]:
        if not tools:
            return None
        
        bedrock_tools = []
        for tool in tools:
            bedrock_tools.append({
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {"json": tool.parameters}
                }
            })
        return {"tools": bedrock_tools} 
