"""
sentarc-ai: Unified LLM client and types.
"""
from .models import get_model, list_models 
# Ensure models are loaded
MODELS = list_models()

from .types import (
    Role, Message, TextContent, ToolUseContent, ToolResultContent, 
    Context, ModelDef, TokenUsage, StreamEvent, Api
)
from .registry import register_api_provider, get_api_provider, ApiProvider
from .providers import anthropic, openai, google

# Register built-in providers
# The following lines are replaced by dedicated provider classes
# register_api_provider(ApiProvider(api="anthropic", stream=anthropic.stream))
# register_api_provider(ApiProvider(api="openai", stream=openai.stream))
# register_api_provider(ApiProvider(api="google", stream=google.stream))

# Register dedicated provider classes
from .providers.anthropic import AnthropicProvider
from .providers.openai_completions import OpenAIProvider
from .providers.openai_responses import OpenAIResponsesProvider
from .providers.openai_codex import OpenAICodexProvider
from .providers.gemini_cli import GeminiCliProvider
from .providers.google import GoogleProvider

register_api_provider(AnthropicProvider())
register_api_provider(OpenAIProvider())
register_api_provider(OpenAIResponsesProvider())
register_api_provider(OpenAICodexProvider())
register_api_provider(GeminiCliProvider())
register_api_provider(GoogleProvider())

try:
    from .providers.amazon_bedrock import BedrockProvider
    register_api_provider(BedrockProvider())
except ImportError:
    pass # dependent on boto3

try:
    from .providers.google_vertex import GoogleVertexProvider
    register_api_provider(GoogleVertexProvider())
except ImportError:
    pass # dependent on google-cloud-aiplatform

from .utils.validation import validate_tool_call
from .stream import stream_simple, complete_simple, stream, complete

__all__ = [
    "Role", "Message", "TextContent", "ToolUseContent", "ToolResultContent",
    "Context", "ModelDef", "TokenUsage", "StreamEvent", "Api",
    "get_model", "MODELS",
    "stream", "complete", "stream_simple", "complete_simple",
    "register_api_provider", "get_api_provider",
    "validate_tool_call"
]
