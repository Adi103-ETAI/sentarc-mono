"""Minimal: create an Agent, prompt once, print assistant text."""
import asyncio
import sys

from sentarc_agent.agent import Agent
from sentarc_agent.types import AgentOptions
from sentarc_ai.models import get_model


async def main() -> None:
    prompt_text = sys.argv[1] if len(sys.argv) > 1 else "Say hello"

    agent = Agent(AgentOptions(
        initial_state={
            "system_prompt": "You are a concise coding assistant.",
            "model": get_model("google", "gemini-2.5-flash-lite-preview-06-17"),
            "thinking_level": "off",
        }
    ))

    await agent.prompt(prompt_text)

    # Grab the last assistant message
    for msg in reversed(agent.state.messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "assistant":
            content = msg.get("content", []) if isinstance(msg, dict) else getattr(msg, "content", [])
            texts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
            if texts:
                print(texts[-1])
                break


if __name__ == "__main__":
    asyncio.run(main())
