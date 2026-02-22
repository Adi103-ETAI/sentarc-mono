"""
Core loop logic for processing the conversational graphs and yielding asynchronous AgentEvent streams.
Handles low-level execution mapping of Tool calls, Steering queues, and Context updates.
"""

import asyncio
from typing import AsyncGenerator, List, Optional, Any, Callable, Awaitable

from sentarc_ai.stream import stream_simple
from sentarc_ai.types import (
    Context,
    ToolCallContent,
    ToolResultMessage,
    AssistantMessage
)
from sentarc_ai.utils.validation import validate_tool_arguments

from sentarc_agent.types import (
    AgentContext,
    AgentLoopConfig,
    AgentMessage,
    AgentEvent,
    AgentStartEvent,
    AgentEndEvent,
    TurnStartEvent,
    TurnEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    MessageEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolExecutionEndEvent,
    AgentToolResult,
    AgentTool,
    StreamFn
)

async def agent_loop(
    prompts: List[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    stream_fn: Optional[StreamFn] = None
) -> AsyncGenerator[AgentEvent, None]:
    """
    Start an agent loop with a new prompt message.
    The prompt is added to the context and events are emitted for it.
    """
    new_messages: List[AgentMessage] = list(prompts)
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=context.messages + prompts,
        tools=context.tools
    )

    yield AgentStartEvent()
    yield TurnStartEvent()
    
    for prompt in prompts:
        yield MessageStartEvent(message=prompt)
        yield MessageEndEvent(message=prompt)

    async for event in _run_loop(current_context, new_messages, config, stream_fn):
        yield event


async def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    stream_fn: Optional[StreamFn] = None
) -> AsyncGenerator[AgentEvent, None]:
    """
    Continue an agent loop from the current context without adding a new message.
    Used for retries - context already has user message or tool results.
    """
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    
    last_message = context.messages[-1]
    # Simple check for basic dictionary structure or role attribute
    role = last_message.get("role") if isinstance(last_message, dict) else getattr(last_message, "role", None)
    
    if role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    new_messages: List[AgentMessage] = []
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages),
        tools=context.tools
    )

    yield AgentStartEvent()
    yield TurnStartEvent()

    async for event in _run_loop(current_context, new_messages, config, stream_fn):
        yield event


async def _run_loop(
    current_context: AgentContext,
    new_messages: List[AgentMessage],
    config: AgentLoopConfig,
    stream_fn: Optional[StreamFn] = None
) -> AsyncGenerator[AgentEvent, None]:
    """Main loop logic shared by agent_loop and agent_loop_continue."""
    first_turn = True
    
    pending_messages: List[AgentMessage] = []
    if config.get_steering_messages:
        pending_messages = await config.get_steering_messages()

    while True:
        has_more_tool_calls = True
        steering_after_tools: Optional[List[AgentMessage]] = None

        while has_more_tool_calls or pending_messages:
            if not first_turn:
                yield TurnStartEvent()
            else:
                first_turn = False

            # Process pending messages (inject before next assistant response)
            if pending_messages:
                for message in pending_messages:
                    yield MessageStartEvent(message=message) # type: ignore
                    yield MessageEndEvent(message=message) # type: ignore
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            # Stream assistant response
            message = None
            async for event in _stream_assistant_response(current_context, config, stream_fn):
                if isinstance(event, AssistantMessage):
                    message = event
                else:
                    yield event
                    
            if not message:
                raise RuntimeError("Failed to generate assistant message.")
                
            new_messages.append(message)

            if getattr(message, "stop_reason", None) in ["error", "aborted"]:
                yield TurnEndEvent(message=message, tool_results=[])
                yield AgentEndEvent(messages=new_messages)
                return

            # Check for tool calls
            tool_calls = [c for c in message.content if isinstance(c, ToolCallContent)]
            has_more_tool_calls = len(tool_calls) > 0

            tool_results: List[ToolResultMessage] = []
            if has_more_tool_calls:
                async for event in _execute_tool_calls(
                    current_context.tools, message, config.get_steering_messages
                ):
                    if isinstance(event, tuple):
                        tool_results.extend(event[0])
                        steering_after_tools = event[1]
                    else:
                        yield event

                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            yield TurnEndEvent(message=message, tool_results=tool_results)

            # Get steering messages after turn completes
            if steering_after_tools and len(steering_after_tools) > 0:
                pending_messages = steering_after_tools
                steering_after_tools = None
            elif config.get_steering_messages:
                pending_messages = await config.get_steering_messages()

        # Check for follow-up messages
        follow_up_messages: List[AgentMessage] = []
        if config.get_follow_up_messages:
            follow_up_messages = await config.get_follow_up_messages()
            
        if follow_up_messages:
            pending_messages = follow_up_messages
            continue

        break

    yield AgentEndEvent(messages=new_messages)


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    stream_fn: Optional[StreamFn] = None
) -> AsyncGenerator[AgentEvent | AssistantMessage, None]:
    """Streams an assistant response, yielding AgentEvents and finally the AssistantMessage."""
    messages = context.messages
    if config.transform_context:
        messages = await config.transform_context(messages, None)

    if not config.convert_to_llm:
        raise ValueError("convert_to_llm function is required in AgentLoopConfig")
        
    llm_messages = config.convert_to_llm(messages)
    if asyncio.iscoroutine(llm_messages):
        llm_messages = await llm_messages # type: ignore

    llm_context = Context(
        system_prompt=context.system_prompt,
        messages=llm_messages, # type: ignore
        tools=context.tools # type: ignore
    )

    stream_function = stream_fn or stream_simple

    resolved_api_key = config.api_key
    if config.get_api_key:
        api_key_result = config.get_api_key(config.model.provider)
        if asyncio.iscoroutine(api_key_result):
            resolved_api_key = await api_key_result # type: ignore
        elif api_key_result:
             resolved_api_key = api_key_result # type: ignore

    config.api_key = resolved_api_key

    # Pass the event stream async generator correctly
    response_stream = stream_function(config.model, llm_context, config)

    partial_message = None
    added_partial = False

    async for event in response_stream:
        if event.type == "start":
            partial_message = event.partial
            context.messages.append(partial_message)
            added_partial = True
            yield MessageStartEvent(message=partial_message) # type: ignore
            
        elif event.type in ["text_start", "text_delta", "text_end", 
                            "thinking_start", "thinking_delta", "thinking_end", 
                            "toolcall_start", "toolcall_delta", "toolcall_end"]:
            if partial_message:
                partial_message = event.partial
                context.messages[-1] = partial_message
                yield MessageUpdateEvent(
                    message=partial_message, # type: ignore
                    assistant_message_event=event
                )
                
        elif event.type in ["done", "error"]:
            # Depending on python implementation, `error` event might just emit the finalized error AssitantMessage
            final_message = event.message if event.type == "done" else event.error
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
                yield MessageStartEvent(message=final_message) # type: ignore
            
            yield MessageEndEvent(message=final_message) # type: ignore
            yield final_message # Explicitly return the finalized AssistantMessage
            return


async def _execute_tool_calls(
    tools: Optional[List[AgentTool]],
    assistant_message: AssistantMessage,
    get_steering_messages: Optional[Callable[[], Awaitable[List[AgentMessage]]]]
) -> AsyncGenerator[Any, None]: # Yields AgentEvents, then finally (tool_results, steering_messages)
    tool_calls = [c for c in assistant_message.content if isinstance(c, ToolCallContent)]
    results: List[ToolResultMessage] = []
    steering_messages: Optional[List[AgentMessage]] = None

    for index, tool_call in enumerate(tool_calls):
        tool = next((t for t in (tools or []) if t.name == tool_call.name), None)

        yield ToolExecutionStartEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            args=tool_call.arguments
        )

        result: AgentToolResult
        is_error = False

        try:
            if not tool:
                raise ValueError(f"Tool {tool_call.name} not found")

            # Validate arguments mapping dictionaries natively using json schema utility port
            validated_args = validate_tool_arguments(tool, tool_call)

            def on_update(partial_result: AgentToolResult):
                return ToolExecutionUpdateEvent(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    args=tool_call.arguments,
                    partial_result=partial_result
                )
            # In purely async python, bridging sync callbacks to event yielders can be tricky.
            # We skip intermediate partial yielding inside the execute function here for simplicity 
            # unless the tool specifically queues it.

            exec_res = tool.execute(tool_call.id, validated_args, None, None)
            if asyncio.iscoroutine(exec_res):
                result = await exec_res
            else:
                result = exec_res # type: ignore

        except Exception as e:
            result = AgentToolResult(
                content=[TextContent(type="text", text=str(e))],
                details={}
            )
            is_error = True

        yield ToolExecutionEndEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=result,
            is_error=is_error
        )

        import time
        tool_result_message = ToolResultMessage(
            role="tool",
            tool_call_id=tool_call.id,
            content=result.content,
            is_error=is_error,
            timestamp=int(time.time() * 1000)
        )

        results.append(tool_result_message)
        yield MessageStartEvent(message=tool_result_message) # type: ignore
        yield MessageEndEvent(message=tool_result_message) # type: ignore

        # Check for steering mid-tool executions
        if get_steering_messages:
            steering = await get_steering_messages()
            if steering:
                steering_messages = steering
                remaining_calls = tool_calls[index + 1:]
                for skipped in remaining_calls:
                    skipped_res = _skip_tool_call(skipped)
                    for event in skipped_res[:-1]:
                        yield event
                    results.append(skipped_res[-1]) # type: ignore
                break

    yield (results, steering_messages)


def _skip_tool_call(tool_call: ToolCallContent) -> tuple:
    result = AgentToolResult(
        content=[TextContent(type="text", text="Skipped due to queued user message.")],
        details={}
    )
    
    events = [
        ToolExecutionStartEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            args=tool_call.arguments
        ),
        ToolExecutionEndEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            result=result,
            is_error=True
        )
    ]
    
    import time
    tool_result_message = ToolResultMessage(
        role="tool",
        tool_call_id=tool_call.id,
        content=result.content,
        is_error=True,
        timestamp=int(time.time() * 1000)
    )
    
    events.extend([
        MessageStartEvent(message=tool_result_message), # type: ignore
        MessageEndEvent(message=tool_result_message) # type: ignore
    ])
    
    events.append(tool_result_message)
    return tuple(events)
