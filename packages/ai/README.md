# sentarc-ai

Unified LLM API with automatic model discovery, provider configuration, token and cost tracking, and simple context persistence and hand-off between models mid-session.

**Note**: This library focuses on models that support tool calling (function calling), as this is essential for agentic workflows.

## Table of Contents

- [Supported Providers](#supported-providers)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Tools](#tools)
  - [Defining Tools](#defining-tools)
  - [Handling Tool Calls](#handling-tool-calls)
  - [Streaming Tool Calls with Partial JSON](#streaming-tool-calls-with-partial-json)
  - [Validating Tool Arguments](#validating-tool-arguments)
  - [Complete Event Reference](#complete-event-reference)
- [Image Input](#image-input)
- [Thinking/Reasoning](#thinkingreasoning)
  - [Unified Interface](#unified-interface-stream_simplecomplete_simple)
  - [Provider-Specific Options (stream/complete)](#provider-specific-options-streamcomplete)
  - [Streaming Thinking Content](#streaming-thinking-content)
- [Stop Reasons](#stop-reasons)
- [APIs, Models, and Providers](#apis-models-and-providers)
  - [Querying Models](#querying-models)
  - [Custom Models](#custom-models)
- [Cross-Provider Handoffs](#cross-provider-handoffs)
- [Context Serialization](#context-serialization)
- [Environment Variables](#environment-variables)
- [OAuth & Advanced Providers](#oauth--advanced-providers)
- [Type Safety](#type-safety)
- [License](#license)

## Supported Providers

- **OpenAI (Completions)**: Standard Chat Completions API (GPT-4o, etc).
- **OpenAI (Responses)**: o-series models (`o1`, `o3`) with reasoning effort.
- **OpenAI (Codex)**: Custom completions endpoints.
- **Anthropic**: Claude 3/3.5 standard models and thinking logs.
- **Google**: Gemini API via `google-generativeai`.
- **Google Cloud Code Assist (Gemini CLI)**: Free-tier Gemini 2.0/3.0 via GCP.
- **Google Vertex AI**: Gemini via Google Cloud ADC.
- **Amazon Bedrock**: Converse Stream API (Claude via AWS).
- **Any OpenAI-compatible API**: Ollama, vLLM, LM Studio, Mistral, Groq, DeepSeek, etc.

## Installation

```bash
pip install -e packages/ai
```

## Quick Start

```python
import asyncio
from sentarc_ai import resolve_model, stream, complete
from sentarc_ai.types import Context, Message, Role, Tool

async def main():
    # Fully typed model resolution (provider, model_id)
    model = resolve_model("openai", "gpt-4o-mini")

    # Define tools using standard JSON Schema style dictionaries
    tools = [
        Tool(
            name="get_time",
            description="Get the current time",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Optional timezone (e.g., America/New_York)"
                    }
                }
            }
        )
    ]

    # Build a conversation context (easily serializable and transferable)
    context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[Message(role=Role.USER, content="What time is it in London?")],
        tools=tools
    )

    # Option 1: Streaming with all event types
    async for event in stream(model, context):
        if event.type == "text_start":
            print("\n[Text started]")
        elif event.type == "text_delta":
            print(event.text, end="", flush=True)
        elif event.type == "thinking_delta":
            print(event.thinking, end="", flush=True)
        elif event.type == "tool_use_start":
            print(f"\n[Tool call started: {event.name}]")
        elif event.type == "tool_use_delta":
            # Partial tool arguments stream as json chunks
            pass
        elif event.type == "tool_use_end":
            print(f"\nTool called: {event.tool_use.name}")
            print(f"Arguments: {event.tool_use.arguments}")
        elif event.type == "stop":
            print(f"\nFinished: {event.stop_reason}")
            print(f"Usage: {event.usage.summary()}")
        elif event.type == "error":
            print(f"\nError: {event.error}")

    # Option 2: Get complete response without streaming
    response_text, tool_calls = await complete(model, context)
    print("Response:", response_text)
    for call in tool_calls:
        print(f"Tool {call.name} requested with args: {call.arguments}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Tools

Tools enable LLMs to interact with external systems. 

### Defining Tools

```python
from sentarc_ai.types import Tool

weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name or coordinates"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    }
)
```

### Handling Tool Calls

Tool results use specific content blocks and are injected back into the context using the `ToolResultMessage`.

```python
from sentarc_ai.types import ToolResultMessage, TextContent

_, tool_calls = await complete(model, context)

for call in tool_calls:
    if call.name == "get_weather":
        # Execute your tool
        result = "{'temp': 22, 'condition': 'Sunny'}"

        # Add tool result with text content
        context.messages.append(ToolResultMessage(
            tool_call_id=call.id,
            content=[TextContent(text=result)],
            is_error=False
        ))

# Continue the conversation so the model can read the tool result
continuation_text, _ = await complete(model, context)
```

### Streaming Tool Calls with Partial JSON

During streaming, tool call arguments are progressively streamed back to you. The `tool_use_delta` event gives you raw JSON string chunks. 

*(Unlike the TS version, the Python port handles `tool_use_delta` strictly as a string chunk. Parsing partial JSON for UI updates must be done application-side using libraries like `partial-json`.)*

### Validating Tool Arguments

The library includes a utility `validate_tool_call` to automatically validate generated tool arguments against your tool's JSON schema using `jsonschema`.

If validation fails, the error can be returned directly to the model as a tool result, allowing it to retry execution.

```python
from sentarc_ai import stream, validate_tool_call
from jsonschema import ValidationError

async for event in stream(model, context):
    if event.type == "tool_use_end":
        try:
            # Validates against the tool's JSON schema
            validated_args = validate_tool_call(tools, event.tool_use)
            
            # Execute your tool logic here
            result = my_tool_logic(validated_args)
            
            context.messages.append(ToolResultMessage(
                tool_call_id=event.tool_use.id,
                content=[TextContent(text=str(result))],
                is_error=False
            ))
            
        except ValidationError as e:
            # Validation failed - return the schema error to the model
            context.messages.append(ToolResultMessage(
                tool_call_id=event.tool_use.id,
                content=[TextContent(text=str(e))],
                is_error=True
            ))
```

### Complete Event Reference

All streaming events emitted during assistant message generation (inherited from `types.StreamEvent`):

| Event Type | Description | Key Properties |
|------------|-------------|----------------|
| `start` | Stream begins | `model`: Current model ID |
| `text_start` | Text block starts | Emitted before text chunks begin |
| `text_delta` | Text chunk received | `text`: New text chunk delta |
| `text_end` | Text block complete | `text`: Final full accumulated text |
| `thinking_start` | Thinking block starts | Emitted before reasoning chunks begin |
| `thinking_delta` | Thinking chunk received | `thinking`: New thinking chunk delta |
| `thinking_end` | Thinking block complete | `thinking`: Final full accumulated thinking string |
| `tool_use_start` | Tool call begins | `id`: Tool identifier, `name`: Function name being called |
| `tool_use_delta` | Tool arguments streaming | `id`: Tool identifier, `partial_input`: JSON string chunk |
| `tool_use_end` | Tool call complete | `tool_use`: Complete parsed `ToolUseContent` object with `id`, `name`, `arguments` |
| `stop` | Stream complete | `stop_reason`: Stop reason ("stop", "length", "tool_use", "content_filter"), `usage`: `TokenUsage` object |
| `error` | Error occurred | `error`: String describing the error type and message |

## Image Input

Models with vision capabilities can process images. You can check if a model supports images via the `model.supports_vision` property.

```python
import base64
from sentarc_ai.types import Message, Role, ImageContent, TextContent

with open("image.jpg", "rb") as f:
    base64_img = base64.b64encode(f.read()).decode('utf-8')

context.messages.append(Message(
    role=Role.USER,
    content=[
        TextContent(text="What is in this image?"),
        ImageContent(media_type="image/jpeg", data=base64_img)
    ]
))
```

## Thinking/Reasoning

Many models support thinking/reasoning capabilities (Claude Sonnet 3.7+, OpenAI o1/o3, DeepSeek R1). 

### Unified Interface (stream_simple/complete_simple)

```python
from sentarc_ai import resolve_model, stream_simple, complete_simple

# Many models across providers support thinking/reasoning natively
model = resolve_model("anthropic", "claude-3-7-sonnet-20250219")
# or resolve_model("openai", "o3-mini")

# Use the simplified reasoning wrapper without needing provider-specific configurations
response_text, tool_calls = await complete_simple(model, context, reasoning="medium")

# For streaming with simple wrappers:
async for event in stream_simple(model, context, reasoning="high"):
    if event.type == "thinking_delta":
        print(event.thinking, end="")
```

### Provider-Specific Options (stream/complete)

You can pass `thinking=True` to the global `stream` function, or control reasoning effort natively using the `StreamOptions` class.

```python
from sentarc_ai.types import StreamOptions, ReasoningEffort

# OpenAI Reasoning (o1, o3)
options = StreamOptions(
    reasoning_effort=ReasoningEffort.HIGH
)

# Anthropic Thinking (Claude 3.7)
options = StreamOptions(
    thinking_enabled=True,
    thinking_budget=8192
)
```

### Streaming Thinking Content

```python
async for event in stream(model, context, thinking=True):
    if event.type == 'thinking_start':
        print('[Model started thinking]')
    elif event.type == 'thinking_delta':
        print(event.thinking, end="")
    elif event.type == 'thinking_end':
        print('\n[Thinking complete]')
```

## Stop Reasons

Every completion ends with a `stop` event indicating how generation ended inside the `stop_reason` field:

- `"stop"` - Normal completion
- `"length"` - Hit maximum token limit
- `"tool_use"` - Model called tools and yielded control
- `"content_filter"` - Output blocked by provider filters

## APIs, Models, and Providers

The library maintains an internal registry mapping `Provider -> API -> File Implementation`. 

### Querying Models

You can iterate registered defaults using the `models` module:

```python
from sentarc_ai.models import list_models, resolve_model

all_models = list_models()
for m in all_models:
    print(f"{m.id} via {m.api} (Context: {m.context_window})")

# Resolve model with fallback creation
# If it's a known provider, it gets structured; if unknown, defaults to generic OpenAI API.
model = resolve_model("ollama", "llama-3.2")
```

### Custom Models

You can dynamically instantiate `ModelDef` for local inference servers or custom proxy endpoints:

```python
from sentarc_ai.types import ModelDef

ollama_model = ModelDef(
    id="llama-3.1-8b",
    provider="ollama",
    api="openai",  # Specifies it should use openai_completions.py
    base_url="http://localhost:11434/v1",
    context_window=128_000,
    supports_thinking=False
)
```

## Cross-Provider Handoffs

The library supports seamless handoffs between different LLM providers within the same conversation through its `transform_messages` pipeline.

- **Thinking Blocks**: If you migrate an Anthropic conversation (which has native `thinking` blocks) to OpenAI GPT-4o, the internal pipeline converts the thinking block to text wrapped in `<thinking>` tags.
- **Tool IDs**: If you hand off a conversation from Mistral to Anthropic, the pipeline handles Tool ID structural differences.
- **Orphaned Results**: Missing Assistant tracking blocks between user tool results are automatically shimmed to prevent 400 Bad Request API errors from strict providers.

## Environment Variables

In Python environments, keys are fetched automatically from `os.environ`:

| Provider | Environment Variable(s) |
|----------|------------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GEMINI_API_KEY` |
| Vertex AI | `GOOGLE_CLOUD_PROJECT` (or `GCLOUD_PROJECT`) + `GOOGLE_CLOUD_LOCATION` |
| Amazon Bedrock | `AWS_REGION` (uses standard Boto3 AWS auth) |

## Context Serialization

The `Context` object, along with all `Message` and `ContentBlock` classes, are structured as standard Python dataclasses. This means they can be easily serialized using the built-in `dataclasses.asdict()` or custom JSON encoders for persistent storage.

```python
import json
from dataclasses import asdict
from sentarc_ai.types import Context, Message, Role

context = Context(messages=[Message(role=Role.USER, content="Hello")])

# Serialize
serialized_context = json.dumps(asdict(context))

# Save to database or file system
with open("chat_history.json", "w") as f:
    f.write(serialized_context)
```

## OAuth & Advanced Providers

Several providers support or require advanced authentication:

- **Google Vertex AI**: Uses Application Default Credentials (ADC). 
    - **Local**: `gcloud auth application-default login`
    - **Production**: Set `GOOGLE_APPLICATION_CREDENTIALS` to a service account JSON.
- **Google Cloud Code Assist (Gemini CLI)**: Uses CLI session tokens. 
- **OpenAI Codex**: Used internally for fine-tuned or organizational models requiring custom `base_url` overrides on the OpenAI standard client.

## Type Safety

The entire `sentarc-ai` package is strictly typed using Python type hints (`from typing import ...`).

- Stream events are a discriminated union of distinct event classes (`TextDeltaEvent`, `ThinkingDeltaEvent`, etc.). 
- Using static type checkers like `mypy` or `pyright` will ensure you handle all possible response structures correctly without runtime crashes.

## License

MIT
