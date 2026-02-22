import pytest
import asyncio

from sentarc_ai.types import (
    AssistantMessage,
    ModelDef,
    TextContent,
    ToolCallContent,
    ToolResultMessage
)
from sentarc_agent.types import (
    AgentEvent,
    AgentMessage,
    AgentState,
    AgentOptions,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult
)
from sentarc_agent.agent import Agent, AgentOptions


# Mock model that returns straight text
class MockModel(ModelDef):
    def __init__(self, override_responses: list = None):
        super().__init__(id="mock", provider="test", api="test")
        self.responses = override_responses or [
            AssistantMessage(
                role="assistant",
                content=[TextContent(type="text", text="Hello back!")],
                api="test",
                provider="test",
                model="mock",
                usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
                stop_reason="stop"
            )
        ]
        self.call_count = 0

async def mock_stream_simple(model: ModelDef, context: dict, options=None, **kwargs):
    if isinstance(model, MockModel):
        res = model.responses[model.call_count]
        model.call_count += 1
        
        events = [
            type("Event", (), {"type": "start", "partial": res})(),
            type("Event", (), {"type": "text_start", "partial": res, "contentIndex": 0})(),
            type("Event", (), {"type": "text_delta", "partial": res, "contentIndex": 0, "delta": "Hello"})(),
            type("Event", (), {"type": "text_end", "partial": res, "contentIndex": 0})(),
            type("Event", (), {"type": "done", "message": res})()
        ]
        
        for e in events:
            yield e
        return
        
    raise ValueError("Not a mock model")


@pytest.mark.asyncio
async def test_agent_initialization():
    agent = Agent()
    assert agent.state.messages == []
    assert not agent.state.is_streaming
    assert agent.state.pending_tool_calls == set()

@pytest.mark.asyncio
async def test_simple_prompt():
    events = []
    
    agent = Agent(AgentOptions(
        stream_fn=mock_stream_simple
    ))
    agent.set_model(MockModel())
    
    agent.subscribe(lambda e: events.append(getattr(e, "type", type(e).__name__)))
    
    await agent.prompt("Hello")
    
    # Assert lifecycle events hit properly
    event_types = [e for e in events]
    assert "agent_start" in event_types
    assert "turn_start" in event_types
    assert "message_start" in event_types  # user prompt
    assert "message_update" in event_types
    assert "message_end" in event_types    # assistant answer
    assert "turn_end" in event_types
    assert "agent_end" in event_types
    
    # Check states
    assert len(agent.state.messages) == 2
    assert agent.state.messages[0]["role"] == "user" # type: ignore
    assert getattr(agent.state.messages[1], "role") == "assistant"
    
@pytest.mark.asyncio
async def test_tool_execution():
    async def mock_tool_execute(tool_call_id, args, signal, on_update):
        return AgentToolResult(
            content=[TextContent(type="text", text=f"Executed with {args['prop']}")],
            details={"prop": args["prop"]}
        )

    tool = AgentTool(
        name="test_tool",
        label="Test Tool",
        description="A test tool",
        parameters={"type": "object", "properties": {"prop": {"type": "string"}}},
        execute=mock_tool_execute
    )
    
    # Mock model will first return a tool call, then a text response
    tool_call_msg = AssistantMessage(
        role="assistant",
        content=[ToolCallContent(type="toolCall", id="call_1", name="test_tool", arguments={"prop": "test_val"})],
        api="test",
        provider="test",
        model="mock",
        usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
        stop_reason="toolUse"
    )
    final_text_msg = AssistantMessage(
        role="assistant",
        content=[TextContent(type="text", text="Done with tool")],
        api="test",
        provider="test",
        model="mock",
        usage={"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "totalTokens": 0},
        stop_reason="stop"
    )
    
    agent = Agent(AgentOptions(stream_fn=mock_stream_simple))
    agent.set_model(MockModel([tool_call_msg, final_text_msg]))
    agent.set_tools([tool])
    
    events = []
    agent.subscribe(lambda e: events.append(getattr(e, "type", type(e).__name__)))
    
    await agent.prompt("Execute tool")
    
    # Agent should have queued up a user message -> assistant toolcall -> toolresult -> assistant text.
    assert "tool_execution_start" in events
    assert "tool_execution_end" in events
    
    messages = agent.state.messages
    assert len(messages) == 4
    
    def _role(m): return m.get("role") if isinstance(m, dict) else getattr(m, "role", None)

    assert _role(messages[0]) == "user"
    assert _role(messages[1]) == "assistant"
    assert _role(messages[2]) == "tool"
    assert _role(messages[3]) == "assistant"

    # Ensure tool execution details mapped into tool result
    assert messages[2].content[0].text == "Executed with test_val" # type: ignore
