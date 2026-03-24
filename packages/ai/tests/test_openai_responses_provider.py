import pytest

from sentarc_ai.providers.openai_responses import OpenAIResponsesProvider
from sentarc_ai.types import (
    Context,
    Message,
    ModelDef,
    ReasoningEffort,
    Role,
    StopEvent,
    StreamOptions,
    TokenUsage,
)


@pytest.mark.asyncio
async def test_openai_responses_uses_options_api_key_and_reasoning(monkeypatch):
    captured = {"api_key": None, "params": None}

    class DummyResponses:
        async def create(self, **params):
            captured["params"] = params

            async def _stream():
                if False:
                    yield None

            return _stream()

    class DummyClient:
        def __init__(self, api_key, base_url=None, default_headers=None):
            captured["api_key"] = api_key
            self.responses = DummyResponses()

    async def fake_process_responses_stream(_stream_resp, _model):
        yield StopEvent(stop_reason="end_turn", usage=TokenUsage())

    monkeypatch.setattr("sentarc_ai.providers.openai_responses.AsyncOpenAI", DummyClient)
    monkeypatch.setattr(
        "sentarc_ai.providers.openai_responses.process_responses_stream",
        fake_process_responses_stream,
    )
    monkeypatch.setattr(
        "sentarc_ai.providers.openai_responses.get_env_api_key",
        lambda _provider: None,
    )

    provider = OpenAIResponsesProvider()
    model = ModelDef(id="o4-mini", provider="openai", api="openai-responses")
    context = Context(messages=[Message(role=Role.USER, content="hello")])
    options = StreamOptions(
        api_key="test-options-key",
        reasoning_effort=ReasoningEffort.HIGH,
        session_id="sess-1",
    )

    events = [e async for e in provider.stream(model, context, options)]

    assert len(events) == 1
    assert events[0].type == "stop"
    assert captured["api_key"] == "test-options-key"
    assert captured["params"]["prompt_cache_key"] == "sess-1"
    assert captured["params"]["reasoning"]["effort"] == "high"
