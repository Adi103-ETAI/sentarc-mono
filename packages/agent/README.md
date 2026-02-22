# sentarc-agent

Stateful agent with tool execution and event streaming. Built on `sentarc-ai`.

## Installation

```bash
pip install -e packages/agent
```

## Quick Start

```python
import asyncio
from sentarc_agent import Agent, AgentOptions
from sentarc_ai.models import get_model

async def main():
    agent = Agent(AgentOptions(
        initial_state={
            "system_prompt": "You are a helpful assistant.",
            "model": get_model("anthropic", "claude-3-5-sonnet-20240620"),
        }
    ))

    events = []
    
    # Subscribe to the event loop directly
    def on_event(event):
        if getattr(event, "type", None) == "message_update":
            if getattr(event.assistant_message_event, "type", None) == "text_delta":
                print(event.assistant_message_event.text, end="", flush=True)

    agent.subscribe(on_event)

    # Prompt the agent
    await agent.prompt("Hello!")

if __name__ == "__main__":
    asyncio.run(main())
```

## Core Concepts

### AgentMessage vs LLM Message

The agent works with `AgentMessage`, a flexible type that can include:
- Standard LLM messages (dictionaries or `Message` objects with `user`, `assistant`, `toolResult` roles).
- Custom app-specific message types via mapping.

LLMs only understand `user`, `assistant`, and `tool`. The `convert_to_llm` function bridges this gap by filtering and transforming messages before each LLM call.

### Message Flow

```
AgentMessage[] → transform_context() → AgentMessage[] → convert_to_llm() → Message[] → LLM
                     (optional)                             (required)
```

1. **`transform_context`**: Prune old messages, inject external context.
2. **`convert_to_llm`**: Filter out UI-only messages, convert custom types to LLM format.

## Event Flow

The agent emits events for UI updates. Understanding the event sequence helps build responsive interfaces.

### `prompt()` Event Sequence

When you call `await agent.prompt("Hello")`:

```
prompt("Hello")
├─ agent_start
├─ turn_start
├─ message_start   { message: userMessage }      # Your prompt
├─ message_end     { message: userMessage }
├─ message_start   { message: assistantMessage } # LLM starts responding
├─ message_update  { message: partial... }       # Streaming chunks
├─ message_update  { message: partial... }
├─ message_end     { message: assistantMessage } # Complete response
├─ turn_end        { message, tool_results: [] }
└─ agent_end       { messages: [...] }
```

### With Tool Calls

If the assistant calls tools, the loop continues automatically:

```
prompt("Read config.json")
├─ agent_start
├─ turn_start
├─ message_start/end     { userMessage }
├─ message_start         { assistantMessage with toolCall }
├─ message_update...
├─ message_end           { assistantMessage }
├─ tool_execution_start  { toolCallId, toolName, args }
├─ tool_execution_update { partialResult }           # If tool streams
├─ tool_execution_end    { toolCallId, result }
├─ message_start/end     { toolResultMessage }
├─ turn_end              { message, tool_results: [toolResult] }
│
├─ turn_start                                        # Next turn
├─ message_start         { assistantMessage }        # LLM responds to tool result
├─ message_update...
├─ message_end
├─ turn_end
└─ agent_end
```

### `continue_session()` Event Sequence

`continue_session()` resumes from existing context without adding a new message. Use it for retries after errors.

```python
# After an error, retry from current state
await agent.continue_session()
```

The last message in context must be `user` or `toolResult` (not `assistant`).

### Event Types

| Event | Description |
|-------|-------------|
| `agent_start` | Agent begins processing |
| `agent_end` | Agent completes with all new messages |
| `turn_start` | New turn begins (one LLM call + tool executions) |
| `turn_end` | Turn completes with assistant message and tool results |
| `message_start` | Any message begins (user, assistant, tool) |
| `message_update` | **Assistant only.** Includes `assistant_message_event` with delta |
| `message_end` | Message completes |
| `tool_execution_start` | Tool begins |
| `tool_execution_update` | Tool streams progress |
| `tool_execution_end` | Tool completes |

## Agent Options

```python
from sentarc_agent import AgentOptions

agent = Agent(AgentOptions(
    # Initial state configurations
    initial_state={
        "system_prompt": "You are a helpful assistant.",
        "model": get_model("openai", "gpt-4o"),
        "thinking_level": "off", # or "minimal", "low", "medium", "high", "xhigh"
        "tools": [my_tool],
        "messages": [],
    },

    # Convert AgentMessage[] to LLM Message[] (required for custom message types)
    convert_to_llm=lambda messages: default_convert_to_llm(messages),

    # Transform context before convert_to_llm (for pruning, compaction)
    transform_context=lambda messages, signal: prune_old_messages(messages),

    # Steering mode: "one-at-a-time" (default) or "all"
    steering_mode="one-at-a-time",

    # Follow-up mode: "one-at-a-time" (default) or "all"
    follow_up_mode="one-at-a-time",

    # Custom stream function wrapper (if needed)
    stream_fn=stream_simple,

    # Session ID for provider caching
    session_id="session-123",

    # Dynamic API key resolution (for expiring tokens)
    get_api_key=get_api_key_async,
    
    # Optional map of token budgets per thinking level
    thinking_budgets={
        "low": 1024,
        "medium": 2048,
        "high": 4096
    }
))
```

## Agent State

```python
from dataclasses import dataclass
from sentarc_agent import AgentState

# Agent State shape
@dataclass
class AgentState:
    system_prompt: str
    model: Optional[ModelDef]
    thinking_level: str
    tools: List[AgentTool]
    messages: List[AgentMessage]
    is_streaming: bool
    stream_message: Optional[AgentMessage]
    pending_tool_calls: Set[str]
    error: Optional[str] = None
```

Access via `agent.state`. During streaming, `stream_message` contains the partial assistant message stream. 

## Methods

### Prompting

```python
# Text prompt
await agent.prompt("Hello")

# With images
from sentarc_ai.types import ImageContent
await agent.prompt("What's in this image?", [
    ImageContent(type="image", data=b"...", mimeType="image/jpeg", source_type="base64")
])

# AgentMessage dictionary directly
import time
await agent.prompt({"role": "user", "content": "Hello", "timestamp": int(time.time() * 1000)})

# Continue from current context (last message must be user or toolResult)
await agent.continue_session()
```

### State Management

```python
agent.set_system_prompt("New prompt")
agent.set_model(get_model("openai", "gpt-4o"))
agent.set_thinking_level("medium")
agent.set_tools([my_tool])
agent.replace_messages(new_messages)
agent.append_message(message)
agent.clear_messages()
agent.reset()  # Clear everything
```

### Session and Thinking Budgets

```python
agent.session_id = "session-123"

agent.thinking_budgets = {
    "minimal": 128,
    "low": 512,
    "medium": 1024,
    "high": 2048
}
```

### Control

```python
agent.abort()                 # Cancel current operation
await agent.wait_for_idle()   # Wait for prompt generation to complete implicitly
```

### Events

```python
unsubscribe = agent.subscribe(lambda event: print(getattr(event, "type", "Unknown Event!")))
unsubscribe()
```

## Tools

Define tools using `AgentTool`:

```python
from sentarc_agent import AgentTool, AgentToolResult
from sentarc_ai.types import TextContent

async def execute_read_file(tool_call_id, params, signal, on_update):
    import os
    if not os.path.exists(params["path"]):
        raise FileNotFoundError(f"File not found: {params['path']}")
        
    with open(params["path"], "r") as f:
        content = f.read()
        
    return AgentToolResult(
        content=[TextContent(type="text", text=content)],
        details={"path": params["path"], "size": len(content)}
    )

read_file_tool = AgentTool(
    name="read_file",
    label="Read File",  # For UI display
    description="Read a file's contents",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}},
    execute=execute_read_file
)

agent.set_tools([read_file_tool])
```

### Error Handling

**Throw an error** when a tool fails. Do not return error messages directly inside content blocks. Thrown exceptions are caught by the agent and inherently mapped to the LLM context with `is_error=True`.

## Steering and Follow-up (Queues)

Interrupt the agent while tools are running (`steer()`), or queue messages to respond once execution finishes (`follow_up()`). 

```python
agent.set_steering_mode("one-at-a-time")
agent.set_follow_up_mode("one-at-a-time")

# While agent is running tools:
agent.steer({
  "role": "user",
  "content": "Stop! Do this instead.",
  "timestamp": int(time.time() * 1000)
})

# Or follow up cleanly after the agent finishes its current work:
agent.follow_up({
  "role": "user",
  "content": "Also summarize the result.",
  "timestamp": int(time.time() * 1000)
})

agent.clear_steering_queue()
agent.clear_follow_up_queue()
agent.clear_all_queues()
```

Use `clear_steering_queue`, `clear_follow_up_queue`, or `clear_all_queues` to drop queued messages dynamically.

When steering messages are detected after a tool completes:
1. Remaining tools are skipped with error results
2. Steering messages are injected
3. LLM responds to the interruption

Follow-up messages are checked only when there are no more tool calls and no steering messages. If any are queued, they are injected and another turn runs.

## Custom Message Types

Unlike TypeScript's rigid declaration merging, Python naturally allows rich `Dict[str, Any]` inheritance under the `AgentMessage` union!

You can define custom message schemas:

```python
# Valid custom AgentMessage (since the typing union supports 'User', 'Assistant', 'Tool', and 'Any dict')
notification_msg = {
    "role": "notification",
    "text": "File updated successfully", 
    "timestamp": int(time.time() * 1000)
}
```

Then, you only need to ensure they are filtered out in your `convert_to_llm` middleware function before the actual provider call:

```python
def my_custom_converter(messages):
    llm_messages = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "notification":
            continue # Filter out custom UI roles
        llm_messages.append(msg)
    return llm_messages

agent = Agent(AgentOptions(convert_to_llm=my_custom_converter))
```

## Low-Level API

For direct control without the stateful `Agent` wrapper, you can interact directly with the asynchronous generator functions parsed inside `agent_loop.py`:

```python
from sentarc_agent import agent_loop, AgentContext, AgentLoopConfig

context = AgentContext(system_prompt="...", messages=[], tools=[])
config = AgentLoopConfig(...)

async for event in agent_loop(messages, context, config):
    print(getattr(event, "type", None))
```

## License
MIT

