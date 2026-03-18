"""Reuse an in-memory session context and continue the conversation."""
import asyncio
import sys

from sentarc_agent.agent import Agent
from sentarc_agent.types import AgentOptions
from sentarc_ai.models import get_model
from sentarc_coding_agent.core.session_manager import SessionManager
from sentarc_coding_agent.core.messages import convert_to_llm


async def main() -> None:
    prompt_text = sys.argv[1] if len(sys.argv) > 1 else "Summarize what we discussed."

    # Build a session with a past assistant exchange
    session = SessionManager.in_memory(cwd=".")
    session.append_message({
        "role": "user",
        "content": [{"type": "text", "text": "Give me three tips for Python readability."}],
    })
    session.append_message({
        "role": "assistant",
        "content": [{"type": "text", "text": "Use type hints, short functions, and clear names."}],
    })

    context = session.build_session_context()

    agent = Agent(AgentOptions(
        convert_to_llm=convert_to_llm,
        initial_state={
            "system_prompt": "Continue the session helpfully.",
            "model": get_model("google", "gemini-2.5-flash-lite-preview-06-17"),
            "thinking_level": "off",
            "messages": context.get("messages", []),
        },
    ))

    await agent.prompt(prompt_text)

    # Save the continuation back into the session
    for msg in agent.state.messages[len(context.get("messages", [])):]:
        session.append_message(msg)  # writes in-memory only

    # Show the most recent assistant reply
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
