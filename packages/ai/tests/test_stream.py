import pytest
import asyncio
from sentarc_ai.stream import stream_simple, complete_simple
from sentarc_ai.types import Context, Message, Role, ModelDef, AssistantMessage, TextEndEvent

class MockProviderEventStream:
    def __init__(self, events):
        self._events = events
        
    def __aiter__(self):
        return self
        
    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

# Patch registry for tests
from sentarc_ai.registry import _registry

class MockApiProvider:
    def stream(self, model, context, options=None):
        return MockProviderEventStream([
            TextEndEvent(text="Result")
        ])

_registry["mock-api"] = MockApiProvider()

@pytest.mark.asyncio
async def test_complete_simple_success():
    model = ModelDef(id="mock", provider="mock", api="mock-api", context_window=100)
    context = Context(messages=[Message(role=Role.USER, content="Hello")])
    
    res = await complete_simple(model, context)
    assert res.role == Role.ASSISTANT
    assert len(res.content) == 1
    assert res.content[0].text == "Result"

@pytest.mark.asyncio
async def test_stream_simple_unregistered_api():
    model = ModelDef(id="mock", provider="mock", api="unknown-api", context_window=100)
    context = Context(messages=[Message(role=Role.USER, content="Hello")])
    
    # Evaluating the generator completely to extract its yield items
    events = [e async for e in stream_simple(model, context)]
    assert len(events) == 1
    assert events[0].type == "error"
    assert "No provider implementation" in events[0].error.error_message
