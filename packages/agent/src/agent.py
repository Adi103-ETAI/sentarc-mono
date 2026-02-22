"""
Stateful Agent class wrapper.
Manages the agent's internal state, tool bindings, event subscriptions, and queues for steering and follow-ups.
Provides `prompt()`, `continue_session()`, `steer()`, and `follow_up()` public accessors.
"""

import asyncio
from typing import List, Optional, Callable, Set, Union, Literal, Awaitable
import time

from sentarc_ai.models import get_model
from sentarc_ai.stream import stream_simple
from sentarc_ai.types import (
    Message,
    ModelDef,
    ImageContent,
    ReasoningEffort
)

from sentarc_agent.types import (
    AgentState,
    AgentOptions,
    AgentMessage,
    AgentEvent,
    AgentTool,
    AgentContext,
    AgentLoopConfig,
    ThinkingLevel,
    StreamFn
)
from sentarc_agent.agent_loop import agent_loop, agent_loop_continue

def default_convert_to_llm(messages: List[AgentMessage]) -> List[Message]:
    """Default converter: Keep only LLM-compatible messages."""
    return [
        m for m in messages if isinstance(m, dict) and m.get("role") in ["user", "assistant", "toolResult"]
        or getattr(m, "role", None) in ["user", "assistant", "toolResult"]
    ] # type: ignore


class Agent:
    def __init__(self, options: Optional[AgentOptions] = None):
        options = options or AgentOptions()
        
        self._state = AgentState(
            system_prompt="",
            model=get_model("google", "gemini-2.5-flash-lite-preview-06-17"),
            thinking_level="off",
            tools=[],
            messages=[],
            is_streaming=False,
            stream_message=None,
            pending_tool_calls=set(),
            error=None
        )
        if options.initial_state:
            for k, v in options.initial_state.items():
                if hasattr(self._state, k):
                    setattr(self._state, k, v)

        self.convert_to_llm = options.convert_to_llm or default_convert_to_llm
        self.transform_context = options.transform_context
        self.steering_mode = options.steering_mode or "one-at-a-time"
        self.follow_up_mode = options.follow_up_mode or "one-at-a-time"
        self.stream_fn = options.stream_fn or stream_simple
        self._session_id = options.session_id
        self._thinking_budgets = options.thinking_budgets
        self.get_api_key = options.get_api_key
        
        self.listeners: Set[Callable[[AgentEvent], None]] = set()
        self.abort_controller: Optional[asyncio.Event] = None
        
        self.steering_queue: List[AgentMessage] = []
        self.follow_up_queue: List[AgentMessage] = []
        self.running_prompt: Optional[asyncio.Task] = None

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @session_id.setter
    def session_id(self, value: Optional[str]):
        self._session_id = value

    @property
    def thinking_budgets(self) -> Optional[Dict[str, int]]:
        return self._thinking_budgets

    @thinking_budgets.setter
    def thinking_budgets(self, value: Optional[Dict[str, int]]):
        self._thinking_budgets = value

    @property
    def state(self) -> AgentState:
        return self._state

    def subscribe(self, fn: Callable[[AgentEvent], None]) -> Callable[[], None]:
        self.listeners.add(fn)
        def unsubscribe():
            self.listeners.discard(fn)
        return unsubscribe

    def set_system_prompt(self, v: str):
        self._state.system_prompt = v

    def set_model(self, m: ModelDef):
        self._state.model = m

    def set_thinking_level(self, l: ThinkingLevel):
        self._state.thinking_level = l

    def set_steering_mode(self, mode: Literal["all", "one-at-a-time"]):
        self.steering_mode = mode

    def set_follow_up_mode(self, mode: Literal["all", "one-at-a-time"]):
        self.follow_up_mode = mode

    def set_tools(self, t: List[AgentTool]):
        self._state.tools = t

    def replace_messages(self, ms: List[AgentMessage]):
        self._state.messages = list(ms)

    def append_message(self, m: AgentMessage):
        self._state.messages.append(m)

    def steer(self, m: AgentMessage):
        """Queue a steering message to interrupt the agent mid-run."""
        self.steering_queue.append(m)

    def follow_up(self, m: AgentMessage):
        """Queue a follow-up message to be processed after the agent finishes."""
        self.follow_up_queue.append(m)

    def clear_steering_queue(self):
        self.steering_queue.clear()

    def clear_follow_up_queue(self):
        self.follow_up_queue.clear()

    def clear_all_queues(self):
        self.steering_queue.clear()
        self.follow_up_queue.clear()

    def has_queued_messages(self) -> bool:
        return len(self.steering_queue) > 0 or len(self.follow_up_queue) > 0

    def _dequeue_steering_messages(self) -> List[AgentMessage]:
        if self.steering_mode == "one-at-a-time":
            if self.steering_queue:
                return [self.steering_queue.pop(0)]
            return []
        steering = list(self.steering_queue)
        self.steering_queue.clear()
        return steering

    def _dequeue_follow_up_messages(self) -> List[AgentMessage]:
        if self.follow_up_mode == "one-at-a-time":
            if self.follow_up_queue:
                return [self.follow_up_queue.pop(0)]
            return []
        follow_up = list(self.follow_up_queue)
        self.follow_up_queue.clear()
        return follow_up

    def clear_messages(self):
        self._state.messages.clear()

    def abort(self):
        if self.abort_controller and not self.abort_controller.is_set():
            self.abort_controller.set()

    async def wait_for_idle(self):
        if self.running_prompt and not self.running_prompt.done():
            await self.running_prompt

    def reset(self):
        self._state.messages.clear()
        self._state.is_streaming = False
        self._state.stream_message = None
        self._state.pending_tool_calls.clear()
        self._state.error = None
        self.clear_all_queues()

    async def prompt(self, input_val: Union[str, AgentMessage, List[AgentMessage]], images: Optional[List[ImageContent]] = None):
        """Send a prompt with an AgentMessage, text, or multiple messages."""
        if self._state.is_streaming:
            raise RuntimeError(
                "Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion."
            )

        model = self._state.model
        if not model:
            raise ValueError("No model configured")

        msgs: List[AgentMessage]

        if isinstance(input_val, list):
            msgs = input_val
        elif isinstance(input_val, str):
            content: List[Union[TextContent, ImageContent]] = [{"type": "text", "text": input_val}] # type: ignore
            if images:
                content.extend(images)
            msgs = [{
                "role": "user",
                "content": content,
                "timestamp": int(time.time() * 1000)
            }] # type: ignore
        else:
            msgs = [input_val]

        await self._run_loop(msgs)

    async def continue_session(self):
        """Continue from current context (used for retries and resuming queued messages)."""
        if self._state.is_streaming:
            raise RuntimeError("Agent is already processing. Wait for completion before continuing.")

        messages = self._state.messages
        if not messages:
            raise ValueError("No messages to continue from")
            
        role = messages[-1].get("role") if isinstance(messages[-1], dict) else getattr(messages[-1], "role", None)
        if role == "assistant":
            queued_steering = self._dequeue_steering_messages()
            if queued_steering:
                await self._run_loop(queued_steering, skip_initial_steering_poll=True)
                return

            queued_follow_up = self._dequeue_follow_up_messages()
            if queued_follow_up:
                await self._run_loop(queued_follow_up)
                return

            raise ValueError("Cannot continue from message role: assistant")

        await self._run_loop(None)

    async def _run_loop(self, messages: Optional[List[AgentMessage]], skip_initial_steering_poll: bool = False):
        model = self._state.model
        if not model:
            raise ValueError("No model configured")

        self.abort_controller = asyncio.Event()
        self._state.is_streaming = True
        self._state.stream_message = None
        self._state.error = None

        reasoning = None if self._state.thinking_level == "off" else self._state.thinking_level

        context = AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=self._state.tools
        )

        async def get_steering_messages() -> List[AgentMessage]:
            nonlocal skip_initial_steering_poll
            if skip_initial_steering_poll:
                skip_initial_steering_poll = False
                return []
            return self._dequeue_steering_messages()

        async def get_follow_up_messages() -> List[AgentMessage]:
            return self._dequeue_follow_up_messages()

        effort = ReasoningEffort(reasoning) if reasoning and reasoning != "none" else ReasoningEffort.NONE

        config = AgentLoopConfig(
            model=model,
            reasoning_effort=effort,
            thinking_enabled=(effort != ReasoningEffort.NONE),
            thinking_budget=10_000 if effort != ReasoningEffort.NONE else 0,
            session_id=self._session_id,
            convert_to_llm=self.convert_to_llm,
            transform_context=self.transform_context,
            get_api_key=self.get_api_key,
            get_steering_messages=get_steering_messages,
            get_follow_up_messages=get_follow_up_messages,
            thinking_budgets=self._thinking_budgets
        )

        partial = None
        
        # We track running_prompt internally
        async def run():
            nonlocal partial
            try:
                stream = agent_loop(messages, context, config, self.stream_fn) if messages else agent_loop_continue(context, config, self.stream_fn)
                
                async for event in stream:
                    # Update internal state based on events
                    event_type = getattr(event, "type", None)
                    
                    if event_type == "message_start":
                        partial = event.message # type: ignore
                        self._state.stream_message = event.message # type: ignore
                    elif event_type == "message_update":
                        partial = event.message # type: ignore
                        self._state.stream_message = event.message # type: ignore
                    elif event_type == "message_end":
                        partial = None
                        self._state.stream_message = None
                        self.append_message(event.message) # type: ignore
                    elif event_type == "tool_execution_start":
                        self._state.pending_tool_calls.add(event.tool_call_id) # type: ignore
                    elif event_type == "tool_execution_end":
                        self._state.pending_tool_calls.discard(event.tool_call_id) # type: ignore
                    elif event_type == "turn_end":
                        if getattr(event.message, "role", None) == "assistant" and getattr(event.message, "error_message", None): # type: ignore
                            self._state.error = getattr(event.message, "error_message", None) # type: ignore
                    elif event_type == "agent_end":
                        self._state.is_streaming = False
                        self._state.stream_message = None
                    
                    self._emit(event)
                    
            except Exception as err:
                from sentarc_ai.types import AssistantMessage
                error_msg = AssistantMessage(
                    role="assistant",
                    content=[TextContent(type="text", text="")], # type: ignore
                    api=model.api, # type: ignore
                    provider=model.provider, # type: ignore
                    model=model.id, # type: ignore
                    usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0}, # type: ignore
                    stop_reason="aborted" if self.abort_controller and self.abort_controller.is_set() else "error",
                    error_message=str(err),
                    timestamp=int(time.time() * 1000)
                )
                self.append_message(error_msg) # type: ignore
                self._state.error = str(err)
                from sentarc_agent.types import AgentEndEvent
                self._emit(AgentEndEvent(messages=[error_msg])) # type: ignore
            finally:
                self._state.is_streaming = False
                self._state.stream_message = None
                self._state.pending_tool_calls.clear()
                self.abort_controller = None

        self.running_prompt = asyncio.create_task(run())
        await self.running_prompt

    def _emit(self, e: AgentEvent):
        for listener in self.listeners:
            listener(e)
