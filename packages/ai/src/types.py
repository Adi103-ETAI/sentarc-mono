"""
Core types for sentarc-ai.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal, Optional


Api = Literal[
    "anthropic",
    "openai",
    "openai-responses", # New
    "google",
    "ollama",
    "mistral",
    "groq",
    # Add others as needed
]

# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

class Role(str, Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"
    TOOL      = "tool"          # tool result going back to the model


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

@dataclass
class TextContent:
    text: str
    type: Literal["text"] = "text"

@dataclass
class ImageContent:
    media_type: str
    data: str  # base64
    type: Literal["image"] = "image"

@dataclass
class ToolCallContent:
    """Represents a tool-use block inside an assistant message."""
    id:        str
    name:      str
    arguments: dict[str, Any]
    type: Literal["tool_call"] = "tool_call"

@dataclass
class ToolResultContent:
    """Represents a tool result block (user turn, fed back to model)."""
    tool_call_id: str
    content:      str
    type: Literal["tool_result"] = "tool_result"

@dataclass
class ThinkingContent:
    thinking: str
    signature: Optional[str] = None # For encrypted reasoning/replay
    type: Literal["thinking"] = "thinking"

@dataclass
class ToolUseContent:
    id:        str
    name:      str
    arguments: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"


# A message content block is one of the above
ContentBlock = TextContent | ThinkingContent | ToolCallContent | ToolResultContent | ImageContent | ToolUseContent


@dataclass
class Message:
    role:    Role
    content: str | list[ContentBlock] = ""       # str shorthand for simple text

    # Convenience: populated on assistant messages that contain tool calls
    tool_calls: list[ToolCallContent] = field(default_factory=list)

    # Populated on tool-result messages
    tool_call_id: Optional[str] = None





# ---------------------------------------------------------------------------
# Tools (schema sent to the model)
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    name:        str
    description: str
    parameters:  dict[str, Any]   # JSON Schema object
    execute:     Optional[Callable] = None

    def to_anthropic(self) -> dict:
        return {
            "name":         self.name,
            "description":  self.description,
            "input_schema": self.parameters,
        }

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name":        self.name,
                "description": self.description,
                "parameters":  self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# Context (what gets sent to the model each turn)
# ---------------------------------------------------------------------------

@dataclass
class Context:
    """
    Contains everything needed for a single LLM call.
    """
    messages:      list[Message]
    system_prompt: Optional[str]       = None
    tools:         list[Tool]          = field(default_factory=list)


# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------

@dataclass
class TextDeltaEvent:
    text: str
    type: Literal["text_delta"] = "text_delta"


@dataclass
class ThinkingDeltaEvent:
    thinking: str
    type: Literal["thinking_delta"] = "thinking_delta"


@dataclass
class ToolCallStartEvent:
    id:   str
    name: str
    type: Literal["tool_call_start"] = "tool_call_start"


@dataclass
class ToolCallDeltaEvent:
    id:            str
    partial_input: str
    type: Literal["tool_call_delta"] = "tool_call_delta"


@dataclass
class ToolCallEndEvent:
    tool_call: ToolCallContent
    type: Literal["tool_call_end"] = "tool_call_end"


@dataclass
class MessageStartEvent:
    type: Literal["message_start"] = "message_start"


@dataclass
class MessageEndEvent:
    usage: "TokenUsage"
    type: Literal["message_end"] = "message_end"


@dataclass
class ErrorEvent:
    error: str
    type: Literal["error"] = "error"


StreamEvent = (
    TextDeltaEvent
    | ThinkingDeltaEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | MessageStartEvent
    | MessageEndEvent
    | ErrorEvent
)


# ---------------------------------------------------------------------------
# Token usage + cost tracking
# ---------------------------------------------------------------------------

@dataclass
class TokenUsage:
    input_tokens:        int   = 0
    output_tokens:       int   = 0
    cache_read_tokens:   int   = 0
    cache_write_tokens:  int   = 0
    cost_usd:            float = 0.0

@dataclass
class AssistantMessage(Message):
    role: Literal[Role.ASSISTANT] = Role.ASSISTANT
    stop_reason: Optional[Literal["stop", "length", "tool_use", "content_filter", "error", "aborted"]] = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    provider: Optional[str] = None
    model:    Optional[str] = None
    api:      Optional[str] = None

@dataclass
class ToolResultMessage(Message):
    role: Literal[Role.TOOL] = Role.TOOL
    tool_call_id: str = ""
    is_error: bool = False
    timestamp: Optional[float] = None

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens       = self.input_tokens        + other.input_tokens,
            output_tokens      = self.output_tokens       + other.output_tokens,
            cache_read_tokens  = self.cache_read_tokens   + other.cache_read_tokens,
            cache_write_tokens = self.cache_write_tokens  + other.cache_write_tokens,
            cost_usd           = self.cost_usd            + other.cost_usd,
        )

    def summary(self) -> str:
        return (
            f"↑{self.input_tokens} ↓{self.output_tokens} "
            f"cache(r={self.cache_read_tokens} w={self.cache_write_tokens}) "
            f"${self.cost_usd:.4f}"
        )


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

@dataclass
class ModelDef:
    """
    Describes a single model: which provider API it speaks,
    context window, capabilities, cost per token.
    """
    id:             str
    provider:       str                  # e.g. "anthropic", "openai", "google", "ollama"
    api:            str                  # underlying API dialect: "anthropic" | "openai" | "google"
    name:           str                  = "" # Display name
    context_window: int                  = 200_000
    max_output:     int                  = 8_192
    supports_tools: bool                 = True
    supports_thinking: bool              = False
    supports_vision:   bool              = False
    input_cost_per_mtok:  float          = 0.0   # USD per million input tokens
    output_cost_per_mtok: float          = 0.0   # USD per million output tokens
    cache_read_cost_per_mtok: float      = 0.0
    cache_write_cost_per_mtok: float     = 0.0
    base_url:       Optional[str]        = None
    extra_headers:  dict[str, str]       = field(default_factory=dict)
    compat:         Optional[OpenAICompletionsCompat] = None


# ---------------------------------------------------------------------------
# New Types (Integration)
# ---------------------------------------------------------------------------

class ReasoningEffort(str, Enum):
    NONE   = "none"
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"
    XHIGH  = "xhigh"  # mapped to high for some providers

# ---------------------------------------------------------------------------
# Compatibility & Routing
# ---------------------------------------------------------------------------

@dataclass
class OpenRouterRouting:
    only: Optional[list[str]] = None
    order: Optional[list[str]] = None

@dataclass
class VercelGatewayRouting:
    only: Optional[list[str]] = None
    order: Optional[list[str]] = None

@dataclass
class OpenAICompletionsCompat:
    supports_store: Optional[bool] = None
    supports_developer_role: Optional[bool] = None
    supports_reasoning_effort: Optional[bool] = None
    supports_usage_in_streaming: Optional[bool] = True
    max_tokens_field: Optional[Literal["max_completion_tokens", "max_tokens"]] = None
    requires_tool_result_name: Optional[bool] = None
    requires_assistant_after_tool_result: Optional[bool] = None
    requires_thinking_as_text: Optional[bool] = None
    requires_mistral_tool_ids: Optional[bool] = None
    thinking_format: Optional[Literal["openai", "zai", "qwen"]] = "openai"
    open_router_routing: Optional[OpenRouterRouting] = None
    vercel_gateway_routing: Optional[VercelGatewayRouting] = None
    supports_strict_mode: Optional[bool] = True


@dataclass
class StreamOptions:
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning_effort: ReasoningEffort = ReasoningEffort.NONE
    thinking_enabled: bool = False
    thinking_budget: Optional[int] = None
    
    # New fields matching TS
    headers: Optional[dict[str, str]] = None
    cache_retention: Literal["none", "short", "long"] = "short"
    session_id: Optional[str] = None


@dataclass
class StartEvent:
    model: str
    type: Literal["start"] = "start"

@dataclass
class TextStartEvent:
    type: Literal["text_start"] = "text_start"

@dataclass
class TextEndEvent:
    text: str
    type: Literal["text_end"] = "text_end"

@dataclass
class ThinkingStartEvent:
    type: Literal["thinking_start"] = "thinking_start"

@dataclass
class ThinkingEndEvent:
    thinking: str
    type: Literal["thinking_end"] = "thinking_end"

@dataclass
class ToolUseStartEvent:
    id:   str
    name: str
    type: Literal["tool_use_start"] = "tool_use_start"

@dataclass
class ToolUseDeltaEvent:
    id:            str
    partial_input: str
    type: Literal["tool_use_delta"] = "tool_use_delta"

@dataclass
class ToolUseEndEvent:
    tool_use: ToolUseContent
    type: Literal["tool_use_end"] = "tool_use_end"

@dataclass
class StopEvent:
    stop_reason: str
    usage: TokenUsage
    type: Literal["stop"] = "stop"

# Update StreamEvent union
StreamEvent = (
    StartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolUseStartEvent
    | ToolUseDeltaEvent
    | ToolUseEndEvent
    | StopEvent
    | ErrorEvent
    # Legacy events (keep for compatibility if needed, or remove if full migration)
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | MessageStartEvent
    | MessageEndEvent
)
