"""
Core types, dataclasses, and event discriminators for the sentarc-agent package.
Provides type safety for the overall agent loop, streaming events, and tool configurations.
"""

from typing import Any, Callable, Dict, List, Literal, Optional, Set, Union, Awaitable
from dataclasses import dataclass, field
import asyncio

from sentarc_ai.types import (
    AssistantMessage,
    StreamEvent,
    Context,
    ImageContent,
    Message,
    ModelDef,
    StreamOptions,
    TextContent,
    Tool,
    ToolCallContent,
    ToolResultMessage,
    ReasoningEffort,
)

# In Python, we can't easily do declaration merging for CustomAgentMessages like in TS.
# We'll rely on `Dict[str, Any]` or a base class `CustomAgentMessage` for users to extend.
# For simplicity and parity with the TS union `Message | CustomAgentMessages[keyof CustomAgentMessages]`,
# we define `AgentMessage` as a union of standard `Message` and `Any` (allowing custom dicts/dataclasses).
AgentMessage = Union[Message, Dict[str, Any]]

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]


@dataclass
class AgentToolResult:
    """Result of an agent tool execution."""
    content: List[Union[TextContent, ImageContent]]
    details: Any


# Callback for streaming tool execution updates
AgentToolUpdateCallback = Callable[[AgentToolResult], None]


@dataclass
class AgentTool(Tool):
    """
    Extends sentarc_ai.types.Tool with an execute function and a label.
    """
    label: str = ""
    
    # execute signature: (tool_call_id: str, args: dict, signal: asyncio.Event, on_update: callable) -> AgentToolResult
    # Using Awaitable here to signify it's an async function
    execute: Callable[[str, Dict[str, Any], Optional[asyncio.Event], Optional[AgentToolUpdateCallback]], Awaitable[AgentToolResult]] = None # type: ignore


@dataclass
class AgentContext:
    system_prompt: str
    messages: List[AgentMessage]
    tools: Optional[List[AgentTool]] = None


@dataclass
class AgentState:
    system_prompt: str
    model: Optional[ModelDef]
    thinking_level: ThinkingLevel
    tools: List[AgentTool]
    messages: List[AgentMessage]
    is_streaming: bool
    stream_message: Optional[AgentMessage]
    pending_tool_calls: Set[str]
    error: Optional[str] = None


@dataclass
class AgentLoopConfig(StreamOptions):
    model: ModelDef = None # type: ignore
    thinking_budgets: Optional[Dict[str, int]] = None
    
    # Converts AgentMessage[] to LLM-compatible Message[] before each LLM call.
    convert_to_llm: Callable[[List[AgentMessage]], Union[List[Message], Awaitable[List[Message]]]] = None # type: ignore

    # Optional transform applied to the context before `convertToLlm`.
    transform_context: Optional[Callable[[List[AgentMessage], Optional[asyncio.Event]], Awaitable[List[AgentMessage]]]] = None

    # Resolves an API key dynamically for each LLM call.
    get_api_key: Optional[Callable[[str], Union[Optional[str], Awaitable[Optional[str]]]]] = None

    # Returns steering messages to inject into the conversation mid-run.
    get_steering_messages: Optional[Callable[[], Awaitable[List[AgentMessage]]]] = None

    # Returns follow-up messages to process after the agent would otherwise stop.
    get_follow_up_messages: Optional[Callable[[], Awaitable[List[AgentMessage]]]] = None


# Events emitted by the Agent for UI updates.

@dataclass
class AgentStartEvent:
    type: Literal["agent_start"] = "agent_start"

@dataclass
class AgentEndEvent:
    messages: List[AgentMessage]
    type: Literal["agent_end"] = "agent_end"

@dataclass
class TurnStartEvent:
    type: Literal["turn_start"] = "turn_start"

@dataclass
class TurnEndEvent:
    message: AgentMessage
    tool_results: List[ToolResultMessage]
    type: Literal["turn_end"] = "turn_end"

@dataclass
class MessageStartEvent:
    message: AgentMessage
    type: Literal["message_start"] = "message_start"

@dataclass
class MessageUpdateEvent:
    message: AgentMessage
    assistant_message_event: StreamEvent
    type: Literal["message_update"] = "message_update"

@dataclass
class MessageEndEvent:
    message: AgentMessage
    type: Literal["message_end"] = "message_end"

@dataclass
class ToolExecutionStartEvent:
    tool_call_id: str
    tool_name: str
    args: Any
    type: Literal["tool_execution_start"] = "tool_execution_start"

@dataclass
class ToolExecutionUpdateEvent:
    tool_call_id: str
    tool_name: str
    args: Any
    partial_result: Any
    type: Literal["tool_execution_update"] = "tool_execution_update"

@dataclass
class ToolExecutionEndEvent:
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool
    type: Literal["tool_execution_end"] = "tool_execution_end"


AgentEvent = Union[
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
]

# A signature for the stream_fn. It aligns with stream_simple from sentarc_ai.stream
StreamFn = Callable[..., Any] # We'll keep this loose in Python to avoid circular deps if needed, passing stream_simple dynamically usually.

@dataclass
class AgentOptions:
    initial_state: Optional[Dict[str, Any]] = None
    convert_to_llm: Optional[Callable[[List[AgentMessage]], Union[List[Message], Awaitable[List[Message]]]]] = None
    transform_context: Optional[Callable[[List[AgentMessage], Optional[asyncio.Event]], Awaitable[List[AgentMessage]]]] = None
    steering_mode: Optional[Literal["all", "one-at-a-time"]] = None
    follow_up_mode: Optional[Literal["all", "one-at-a-time"]] = None
    stream_fn: Optional[StreamFn] = None
    session_id: Optional[str] = None
    get_api_key: Optional[Callable[[str], Union[Optional[str], Awaitable[Optional[str]]]]] = None
    thinking_budgets: Optional[Dict[str, int]] = None
