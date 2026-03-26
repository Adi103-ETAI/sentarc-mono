import pytest
import importlib

from sentarc_coding_agent.core.compaction.compaction import (
    CompactionSettings,
    compact_messages,
)
from sentarc_ai.types import AssistantMessage, ModelDef, TextContent


@pytest.mark.asyncio
async def test_compaction_uses_complete_simple_context_contract(monkeypatch):
    captured = {"called": False, "system_prompt": None, "user_content": None}

    async def fake_complete_simple(model, context, options=None):
        captured["called"] = True
        captured["system_prompt"] = context.system_prompt
        captured["user_content"] = context.messages[0].content
        return AssistantMessage(
            content=[TextContent(text="Structured summary")],
            provider=model.provider,
            model=model.id,
            api=model.api,
            stop_reason="stop",
        )

    stream_module = importlib.import_module("sentarc_ai.stream")
    monkeypatch.setattr(stream_module, "complete_simple", fake_complete_simple)

    entries = [
        {
            "type": "message",
            "id": "m1",
            "parentId": None,
            "timestamp": "2026-01-01T00:00:00Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "task"}]},
        },
        {
            "type": "message",
            "id": "m2",
            "parentId": "m1",
            "timestamp": "2026-01-01T00:00:01Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "working"}]},
        },
        {
            "type": "message",
            "id": "m3",
            "parentId": "m2",
            "timestamp": "2026-01-01T00:00:02Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "continue"}]},
        },
    ]

    model = ModelDef(id="gemini-2.5-flash", provider="google", api="google")
    settings = CompactionSettings(enabled=True, keep_recent_tokens=1)

    result = await compact_messages(entries, model, settings=settings)

    assert captured["called"] is True
    assert captured["system_prompt"]
    assert isinstance(captured["user_content"], str)
    assert result is not None
    assert "Structured summary" in result.summary
