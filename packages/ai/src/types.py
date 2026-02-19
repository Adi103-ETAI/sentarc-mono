"""
Core types for sentarc-ai.
Python port of packages/ai/src/types.ts from sentarc-mono.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal, Optional


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


# A message content block is one of the above
ContentBlock = TextContent | ToolCallContent | ToolResultContent


@dataclass
class Message:
    role:    Role
    content: str | list[ContentBlock]       # str shorthand for simple text

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
    Mirrors sentarc-mono's Context type.
    Contains everything needed for a single LLM call.
    """
    messages:      list[Message]
    system_prompt: Optional[str]       = None
    tools:         list[Tool]          = field(default_factory=list)


# ---------------------------------------------------------------------------
# Streaming events  (mirrors sentarc-mono's AssistantMessageEvent union)
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
# Token usage + cost tracking  (mirrors sentarc-mono's TokenUsage)
# ---------------------------------------------------------------------------

@dataclass
class TokenUsage:
    input_tokens:        int   = 0
    output_tokens:       int   = 0
    cache_read_tokens:   int   = 0
    cache_write_tokens:  int   = 0
    cost_usd:            float = 0.0

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
# Model definition  (mirrors sentarc-mono's Model<TApi>)
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
    context_window: int                  = 200_000
    max_output:     int                  = 8_192
    supports_tools: bool                 = True
    supports_thinking: bool              = False
    input_cost_per_mtok:  float          = 0.0   # USD per million input tokens
    output_cost_per_mtok: float          = 0.0   # USD per million output tokens
    base_url:       Optional[str]        = None
    extra_headers:  dict[str, str]       = field(default_factory=dict)
