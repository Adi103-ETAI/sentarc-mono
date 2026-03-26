import pytest

from sentarc_ai.providers.openai_completions import OpenAIProvider
from sentarc_ai.types import Context, Message, ModelDef, Role, StreamOptions, TokenUsage, StopEvent


@pytest.mark.asyncio
async def test_openai_completions_uses_options_api_key_and_ignores_missing_tool_choice(monkeypatch):
    captured = {"api_key": None}

    class DummyChunk:
        def __init__(self):
            class Choice:
                def __init__(self):
                    class Delta:
                        content = None
                        tool_calls = None
                    self.delta = Delta()
            self.choices = [Choice()]
            self.usage = type("Usage", (), {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})

    class DummyResponseStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if getattr(self, "_done", False):
                raise StopAsyncIteration
            self._done = True
            return DummyChunk()

    class DummyCompletions:
        async def create(self, **kwargs):
            return DummyResponseStream()

    class DummyChat:
        def __init__(self):
            self.completions = DummyCompletions()

    class DummyClient:
        def __init__(self, api_key, base_url=None, default_headers=None):
            captured["api_key"] = api_key
            self.chat = DummyChat()

    monkeypatch.setattr("sentarc_ai.providers.openai_completions.AsyncOpenAI", DummyClient)
    monkeypatch.setattr("sentarc_ai.providers.openai_completions.get_env_api_key", lambda _provider: None)

    provider = OpenAIProvider()
    model = ModelDef(id="gpt-4o", provider="openai", api="openai")
    context = Context(messages=[Message(role=Role.USER, content="hello")])
    options = StreamOptions(api_key="opt-key")

    events = [e async for e in provider.stream(model, context, options)]

    assert captured["api_key"] == "opt-key"
    assert any(getattr(e, "type", None) == "stop" for e in events)
