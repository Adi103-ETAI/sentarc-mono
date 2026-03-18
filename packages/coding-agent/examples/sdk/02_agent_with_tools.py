"""Enable built-in tools and log tool calls."""
import asyncio
import sys

from sentarc_agent.agent import Agent
from sentarc_agent.types import AgentOptions
from sentarc_ai.models import get_model
from sentarc_coding_agent.core.tools import create_tools
from sentarc_coding_agent.core.messages import convert_to_llm


def print_event(event) -> None:
    etype = getattr(event, "type", None)
    if etype == "tool_execution_start":
        print(f"[tool start] {event.tool_name} args={event.args}")
    elif etype == "tool_execution_end":
        print(f"[tool end] {event.tool_name} ok={not getattr(event, 'is_error', False)}")


async def main() -> None:
    prompt_text = sys.argv[1] if len(sys.argv) > 1 else "List the files in the current directory."

    tools = create_tools(cwd=".")

    agent = Agent(AgentOptions(
        convert_to_llm=convert_to_llm,
        initial_state={
            "system_prompt": "You can use tools to inspect the workspace.",
            "model": get_model("google", "gemini-2.5-flash-lite-preview-06-17"),
            "thinking_level": "minimal",
            "tools": tools,
        },
    ))

    agent.subscribe(print_event)
    await agent.prompt(prompt_text)

    # Print the last assistant text
    for msg in reversed(agent.state.messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "assistant":
            content = msg.get("content", []) if isinstance(msg, dict) else getattr(msg, "content", [])
            texts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
            if texts:
                print("\n[assistant]\n" + texts[-1])
                break


if __name__ == "__main__":
    asyncio.run(main())
