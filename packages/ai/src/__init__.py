"""
sentarc_ai — Unified LLM client and types.
Python port of @sentarc-labs/sentarc-ai from sentarc-mono.

Public API (mirrors packages/ai/src/index.ts exports):

    from sentarc_ai import stream, complete, get_model, resolve_model
    from sentarc_ai.types import Message, Role, Tool, Context, TokenUsage
    from sentarc_ai.models import register_model, list_models
"""

from .types import (
    Role,
    Message,
    Tool,
    Context,
    TextContent,
    ToolCallContent,
    ToolResultContent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    ToolCallStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    MessageStartEvent,
    MessageEndEvent,
    ErrorEvent,
    StreamEvent,
    TokenUsage,
    ModelDef,
)

from .models import (
    get_model,
    resolve_model,
    register_model,
    list_models,
    PROVIDER_DEFAULTS,
)

from .stream import stream, complete

__all__ = [
    # Types
    "Role",
    "Message",
    "Tool",
    "Context",
    "TextContent",
    "ToolCallContent",
    "ToolResultContent",
    "TextDeltaEvent",
    "ThinkingDeltaEvent",
    "ToolCallStartEvent",
    "ToolCallDeltaEvent",
    "ToolCallEndEvent",
    "MessageStartEvent",
    "MessageEndEvent",
    "ErrorEvent",
    "StreamEvent",
    "TokenUsage",
    "ModelDef",
    # Model registry
    "get_model",
    "resolve_model",
    "register_model",
    "list_models",
    "PROVIDER_DEFAULTS",
    # Streaming
    "stream",
    "complete",
]
