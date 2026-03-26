import pytest

from sentarc_ai.registry import ApiProvider, get_api_provider, register_api_provider
import sentarc_ai.registry as registry_module


class DummyProvider:
    async def stream(self, model, context, options=None):
        if False:
            yield None


def test_get_api_provider_resolves_aliases():
    original_registry = dict(registry_module._registry)
    try:
        registry_module._registry.clear()
        register_api_provider(ApiProvider(api="openai", stream=DummyProvider().stream))
        register_api_provider(ApiProvider(api="google", stream=DummyProvider().stream))
        register_api_provider(ApiProvider(api="anthropic", stream=DummyProvider().stream))

        assert get_api_provider("openai") is not None
        assert get_api_provider("openai-completions") is not None
        assert get_api_provider("google-generative-ai") is not None
        assert get_api_provider("anthropic-messages") is not None
        assert get_api_provider("unknown-api") is None
    finally:
        registry_module._registry.clear()
        registry_module._registry.update(original_registry)


def test_register_api_provider_validates_stream_callable():
    original_registry = dict(registry_module._registry)
    try:
        registry_module._registry.clear()
        bad = ApiProvider(api="openai", stream=DummyProvider().stream)
        bad.stream = None
        with pytest.raises(TypeError, match="stream must be callable"):
            register_api_provider(bad)
    finally:
        registry_module._registry.clear()
        registry_module._registry.update(original_registry)


def test_register_api_provider_validates_api_name():
    original_registry = dict(registry_module._registry)
    try:
        registry_module._registry.clear()
        bad = ApiProvider(api="openai", stream=DummyProvider().stream)
        bad.api = ""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            register_api_provider(bad)
    finally:
        registry_module._registry.clear()
        registry_module._registry.update(original_registry)
