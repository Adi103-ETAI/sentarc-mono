import pytest

from sentarc_ai.providers.openai_utils import convert_responses_messages, process_responses_stream
from sentarc_ai.types import (
    Context,
    ImageContent,
    Message,
    ModelDef,
    Role,
    TextContent,
    ToolResultMessage,
)


def test_convert_responses_messages_tool_role_and_vision_flag():
    model = ModelDef(id="o4-mini", provider="openai", api="openai-responses", supports_vision=False)
    context = Context(
        system_prompt="You are helpful",
        messages=[
            Message(role=Role.USER, content=[TextContent(text="hello")]),
            ToolResultMessage(
                role=Role.TOOL,
                tool_call_id="call_123|fc_123",
                content=[
                    TextContent(text="done"),
                    ImageContent(media_type="image/png", data="abc"),
                ],
            ),
        ],
    )

    out = convert_responses_messages(model, context, allowed_tool_call_providers={"openai"})

    assert out[0]["role"] == "system"
    tool_outputs = [m for m in out if isinstance(m, dict) and m.get("type") == "function_call_output"]
    assert len(tool_outputs) == 1
    assert tool_outputs[0]["call_id"] == "call_123"
    assert tool_outputs[0]["output"] == "done"
    # supports_vision=False should avoid appending extra input_image user content
    assert not any(
        isinstance(m, dict)
        and m.get("role") == "user"
        and any(isinstance(p, dict) and p.get("type") == "input_image" for p in m.get("content", []))
        for m in out
    )


@pytest.mark.asyncio
async def test_process_responses_stream_completed_usage_without_total_tokens_field():
    async def fake_stream():
        yield {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 7,
                    "input_tokens_details": {"cached_tokens": 3},
                },
            },
        }

    model = ModelDef(id="o4-mini", provider="openai", api="openai-responses")
    events = [e async for e in process_responses_stream(fake_stream(), model)]

    assert events[0].type == "start"
    assert events[-1].type == "stop"
    assert events[-1].usage.input_tokens == 12
    assert events[-1].usage.output_tokens == 7
    assert events[-1].usage.cache_read_tokens == 3
