"""
API provider registry.
"""
from __future__ import annotations
from typing import Any, Optional, Protocol

from .types import ModelDef, Context, StreamEvent, ReasoningEffort, Api
from .providers.openai_completions import OpenAIProvider
from .providers.openai_responses import OpenAIResponsesProvider
from .providers.openai_codex import OpenAICodexProvider
from .providers.gemini_cli import GeminiCliProvider
from .providers.google import GoogleProvider
from .providers.amazon_bedrock import BedrockProvider
from .providers.google_vertex import GoogleVertexProvider

# Define protocols for stream functions
class StreamFunction(Protocol):
    def __call__(
        self,
        model: ModelDef,
        context: Context,
        reasoning: ReasoningEffort = ReasoningEffort.NONE,
    ) -> Any: # Returns AsyncIterator[StreamEvent]
        ...

class ApiProvider:
    def __init__(self, api: Api, stream: StreamFunction, stream_simple: Optional[StreamFunction] = None):
        self.api = api
        self.stream = stream
        self.stream_simple = stream_simple or stream

_registry: dict[str, ApiProvider] = {}

_API_ALIASES: dict[str, str] = {
    # Generated/static model catalog aliases
    "anthropic-messages": "anthropic",
    "openai-completions": "openai",
    "openai-chat-completions": "openai",
    "google-generative-ai": "google",
}

def register_api_provider(provider: ApiProvider) -> None:
    """Register an API provider."""
    if not getattr(provider, "api", None):
        raise ValueError("ApiProvider.api must be a non-empty string")
    if not callable(getattr(provider, "stream", None)):
        raise TypeError("ApiProvider.stream must be callable")
    if getattr(provider, "stream_simple", None) is not None and not callable(provider.stream_simple):
        raise TypeError("ApiProvider.stream_simple must be callable when provided")

    _registry[provider.api] = provider

def get_api_provider(api: str) -> Optional[ApiProvider]:
    """Get a registered API provider."""
    provider = _registry.get(api)
    if provider:
        return provider

    canonical_api = _API_ALIASES.get(api)
    if canonical_api:
        return _registry.get(canonical_api)

    return None

def clear_api_providers() -> None:
    """Clear all registered providers."""
    _registry.clear()
