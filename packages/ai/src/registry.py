"""
API provider registry.
"""
from __future__ import annotations
from typing import Callable, Any, Optional, Protocol, TypeVar

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

def register_api_provider(provider: ApiProvider) -> None:
    """Register an API provider."""
    # sentarc-mono logic checks for specific API type mismatch, handled here by simple map
    _registry[provider.api] = provider

def get_api_provider(api: str) -> Optional[ApiProvider]:
    """Get a registered API provider."""
    return _registry.get(api)

def clear_api_providers() -> None:
    """Clear all registered providers."""
    _registry.clear()
