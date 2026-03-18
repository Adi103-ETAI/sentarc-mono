<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License" />
</p>

# Sentarc Monorepo

> **Looking for the arc coding agent?** See **[packages/coding-agent](packages/coding-agent)** for installation and usage.

Tools for building AI agents with a unified multi-provider LLM API.

## Packages

| Package | Description |
|---------|-------------|
| **[sentarc-ai](packages/ai)** | Unified multi-provider LLM API (OpenAI, Anthropic, Google, AWS Bedrock, etc.) |
| **[sentarc-agent](packages/agent)** | Stateful agent runtime with tool calling and event streaming |
| **[sentarc-tui](packages/tui)** | Terminal UI library built on textual with differential rendering |
| **[sentarc-coding-agent](packages/coding-agent)** | Interactive coding agent CLI (`arc` command) |

## Supported LLM Providers

- **OpenAI** - GPT-4, GPT-4o, o1, o3, etc.
- **Anthropic** - Claude 3.5, Claude 4, etc.
- **Google** - Gemini API
- **Google Cloud** - Vertex AI, Code Assist
- **AWS** - Bedrock (Claude, Titan, etc.)
- **OpenAI-Compatible** - Ollama, vLLM, Mistral, Groq, DeepSeek, and more

## Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages in editable mode
pip install -e packages/ai
pip install -e packages/agent
pip install -e packages/tui
pip install -e packages/coding-agent

# Run the coding agent
arc
```

## Scripts

| Script | Description |
|--------|-------------|
| `./test.sh` | Run all tests (skips LLM-dependent tests without API keys) |
| `./arc-test.sh` | Run arc from sources (must be run from repo root) |

```bash
# Run tests
./test.sh                     # Run all tests without API keys
./test.sh -v                  # Verbose output

# Run arc from sources
./arc-test.sh                 # Run arc normally
./arc-test.sh --no-env        # Run without API keys
```

## License

MIT
