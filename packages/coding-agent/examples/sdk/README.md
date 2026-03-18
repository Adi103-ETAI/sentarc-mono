# SDK examples

Small, runnable Python snippets that show how to drive the agent and tools programmatically.

## Prerequisites
- Install editable deps: `pip install -e packages/ai -e packages/agent -e packages/coding-agent`
- Set a model API key: e.g. `export ANTHROPIC_API_KEY=...` or `OPENAI_API_KEY=...`

## Examples
- `01_minimal_agent.py` — send a prompt and print the assistant text.
- `02_agent_with_tools.py` — enable built-in tools and log tool calls.
- `03_session_resume.py` — reuse a session history in-memory, then continue it.

Run with:
```bash
python packages/coding-agent/examples/sdk/01_minimal_agent.py "Say hi"
```
