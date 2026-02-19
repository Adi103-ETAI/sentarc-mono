"""
Core agent loop.
Python port of @sentarc-labs/sentarc-agent-core from sentarc-mono.

Responsibilities:
  - Manage conversation history (messages list)
  - Drive the tool-call loop (prompt → LLM → tool → repeat)
  - Fire lifecycle events via the event system
  - Track token usage across the session
"""
from __future__ import annotations
import asyncio
from typing import Any, AsyncIterator, Callable, Optional

from sentarc_ai import (
    stream, complete,
    resolve_model,
    Message, Role, Context, Tool,
    ToolCallContent, ToolResultContent,
    TextDeltaEvent, ToolCallEndEvent, MessageEndEvent,
    TokenUsage, ModelDef, StreamEvent,
)


# ---------------------------------------------------------------------------
# Simple event bus  (mirrors sentarc-mono's on/emit pattern in AgentSession)
# ---------------------------------------------------------------------------

LIFECYCLE_EVENTS = [
    "session_start",
    "session_end",
    "turn_start",
    "turn_end",
    "tool_call",            # fired before tool executes — handler can block
    "tool_result",          # fired after tool executes — handler can modify result
    "context",              # fired before each LLM call — handler can modify context
    "message_end",          # fired after each LLM response
]


class EventBus:
    def __init__(self):
        self._handlers: dict[str, list[Callable]] = {e: [] for e in LIFECYCLE_EVENTS}

    def on(self, event: str, handler: Callable) -> None:
        if event not in self._handlers:
            raise ValueError(f"Unknown event: {event!r}. Valid: {LIFECYCLE_EVENTS}")
        self._handlers[event].append(handler)

    def off(self, event: str, handler: Callable) -> None:
        self._handlers.get(event, []).remove(handler)

    async def emit(self, event: str, data: Any = None, *extra) -> Any:
        for h in self._handlers.get(event, []):
            try:
                result = await h(data, *extra) if asyncio.iscoroutinefunction(h) else h(data, *extra)
                if result is not None:
                    data = result
            except Exception as e:
                print(f"[EventBus error: {event}] {e}")
        return data


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """Register and execute tools. Mirrors coding-agent's tool system."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    async def execute(self, tool_call: ToolCallContent) -> str:
        tool = self._tools.get(tool_call.name)
        if not tool:
            return f"Error: unknown tool '{tool_call.name}'"
        if not tool.execute:
            return f"Error: tool '{tool_call.name}' has no execute function"
        try:
            if asyncio.iscoroutinefunction(tool.execute):
                result = await tool.execute(**tool_call.arguments)
            else:
                result = tool.execute(**tool_call.arguments)
            return str(result)
        except Exception as e:
            return f"Tool error [{tool_call.name}]: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Agent session  (mirrors AgentSession in sentarc-mono)
# ---------------------------------------------------------------------------

class AgentSession:
    """
    Manages a single agent conversation session.

    Usage:
        session = AgentSession(provider="anthropic")
        result  = await session.prompt("List the .py files here")
    """

    def __init__(
        self,
        provider:      str            = "anthropic",
        model_id:      Optional[str]  = None,
        system_prompt: Optional[str]  = None,
        max_turns:     int            = 50,
    ):
        self.model:        ModelDef     = resolve_model(provider, model_id)
        self.system_prompt: str         = system_prompt or _DEFAULT_SYSTEM
        self.max_turns:    int          = max_turns
        self.messages:     list[Message] = []
        self.tools:        ToolRegistry = ToolRegistry()
        self.events:       EventBus     = EventBus()
        self.total_usage:  TokenUsage   = TokenUsage()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def prompt(
        self,
        user_message: str,
        on_event: Optional[Callable[[StreamEvent], None]] = None,
        thinking: bool = False,
    ) -> str:
        """
        Send a user message, run the agentic loop, return final text.
        Mirrors AgentSession.prompt() in sentarc-mono.
        """
        self.messages.append(Message(role=Role.USER, content=user_message))
        await self.events.emit("session_start", user_message)

        final_text = ""

        for turn in range(self.max_turns):
            await self.events.emit("turn_start", turn + 1)

            context = Context(
                messages=self._build_context_messages(),
                system_prompt=self.system_prompt,
                tools=self.tools.all(),
            )
            # Allow hooks to modify context before LLM call
            context = await self.events.emit("context", context) or context

            text_parts: list[str]          = []
            pending_tcs: list[ToolCallContent] = []

            async for event in stream(self.model, context, thinking=thinking):
                if on_event:
                    on_event(event)

                if isinstance(event, TextDeltaEvent):
                    text_parts.append(event.text)

                elif isinstance(event, ToolCallEndEvent):
                    blocked = await self.events.emit("tool_call", event.tool_call)
                    if not blocked:
                        pending_tcs.append(event.tool_call)

                elif isinstance(event, MessageEndEvent):
                    self.total_usage = self.total_usage + event.usage
                    await self.events.emit("message_end", event.usage)

            # Record assistant turn
            assistant_text = "".join(text_parts)
            asst_msg = Message(
                role=Role.ASSISTANT,
                content=assistant_text,
                tool_calls=pending_tcs,
            )
            self.messages.append(asst_msg)

            # If no tool calls → done
            if not pending_tcs:
                final_text = assistant_text
                break

            # Execute tools and collect results
            for tc in pending_tcs:
                raw_result = await self.tools.execute(tc)
                result     = await self.events.emit("tool_result", raw_result, tc) or raw_result
                self.messages.append(Message(
                    role=Role.TOOL,
                    content=str(result),
                    tool_call_id=tc.id,
                ))

            await self.events.emit("turn_end", turn + 1)

        await self.events.emit("session_end", final_text)
        return final_text

    def set_model(self, provider: str, model_id: Optional[str] = None) -> None:
        """Switch models mid-session. Mirrors AgentSession.setModel()."""
        self.model = resolve_model(provider, model_id)

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()

    def usage_summary(self) -> str:
        return self.total_usage.summary()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context_messages(self) -> list[Message]:
        """
        Return messages ready for the LLM.
        System prompt is passed separately via Context.system_prompt.
        """
        return [m for m in self.messages if m.role != Role.SYSTEM]


# ---------------------------------------------------------------------------
# Default system prompt
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM = (
    "You are a helpful AI assistant. "
    "When tools are available, use them to complete tasks accurately. "
    "Think step by step before acting."
)
