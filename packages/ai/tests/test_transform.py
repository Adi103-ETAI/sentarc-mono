import pytest
from sentarc_ai.types import (
    Message, Role, TextContent, ThinkingContent, ToolCallContent,
    ToolResultContent, AssistantMessage, ModelDef, ToolUseContent, ToolResultMessage
)
from sentarc_ai.transform_messages import transform_messages

def test_transform_messages_sanitizes_text():
    # Model definition
    model = ModelDef(id="claude-3", provider="anthropic", api="anthropic-messages", context_window=100_000, max_output=10_000)
    
    # Text input with an isolated surrogate
    user_msg_str = Message(role=Role.USER, content="Bad \ud83d string")
    user_msg_blocks = Message(role=Role.USER, content=[TextContent(text="Bad \ud83d block")])
    assistant_msg = AssistantMessage(
        role=Role.ASSISTANT, 
        content=[TextContent(text="Bad \ude48 response")],
        provider="anthropic",
        model="claude-3"
    )

    transformed = transform_messages([user_msg_str, user_msg_blocks, assistant_msg], model)

    assert len(transformed) == 3
    assert transformed[0].content == "Bad  string"
    assert transformed[1].content[0].text == "Bad  block"
    assert transformed[2].content[0].text == "Bad  response"

def test_transform_messages_handles_handoff_thinking():
    model_b = ModelDef(id="gpt-4o", provider="openai", api="openai-responses", context_window=100_000, max_output=10_000)

    # Generated from an Anthropic model
    assistant_msg = AssistantMessage(
        role=Role.ASSISTANT, 
        content=[
            ThinkingContent(thinking="I need to \ud83d think", signature="sig_123"),
            TextContent(text="Final answer")
        ],
        provider="anthropic",
        model="claude-3-sonnet"
    )

    transformed = transform_messages([assistant_msg], model_b)
    
    assert len(transformed) == 1
    content = transformed[0].content
    
    # The thinking block should be converted to a TextContent block because provider/models differ
    assert len(content) == 2
    assert isinstance(content[0], TextContent)
    assert content[0].text == "I need to  think"  # Note the sanitized surrogate!
    assert isinstance(content[1], TextContent)
    assert content[1].text == "Final answer"

def test_transform_messages_same_model_thinking():
    model = ModelDef(id="claude-3", provider="anthropic", api="anthropic-messages", context_window=100_000, max_output=10_000)

    assistant_msg = AssistantMessage(
        role=Role.ASSISTANT, 
        content=[
            ThinkingContent(thinking="Still thinking", signature="sig_123")
        ],
        provider="anthropic",
        model="claude-3"
    )

    transformed = transform_messages([assistant_msg], model)
    
    # Should stay as a ThinkingContent block
    assert isinstance(transformed[0].content[0], ThinkingContent)
    assert transformed[0].content[0].thinking == "Still thinking"
    assert transformed[0].content[0].signature == "sig_123"

def test_transform_messages_orphaned_tool_calls():
    model = ModelDef(id="claude-3", provider="anthropic", api="anthropic-messages", context_window=100_000, max_output=10_000)

    # An assistant message that made a tool call but never got a tool result
    tool_use = ToolCallContent(id="call_x", name="get_weather", arguments={"loc": "nyc"})
    assistant_msg = AssistantMessage(
        role=Role.ASSISTANT, 
        content=[tool_use],
        tool_calls=[tool_use], # The stream handler builds this normally
        provider="anthropic",
        model="claude-3"
    )

    # A user message abruptly follows without a toolResult message
    user_msg = Message(role=Role.USER, content="Forget that")

    transformed = transform_messages([assistant_msg, user_msg], model)

    # It should dynamically insert a ToolResultMessage for the orphan BEFORE the user message
    assert len(transformed) == 3
    assert transformed[0] == assistant_msg
    assert isinstance(transformed[1], ToolResultMessage)
    assert transformed[1].tool_call_id == "call_x"
    assert transformed[1].is_error is True
    assert transformed[2] == user_msg
