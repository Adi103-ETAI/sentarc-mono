#!/usr/bin/env bash
# Run tests for sentarc-mono (skips LLM-dependent tests without API keys)
# Usage: ./test.sh [pytest options]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

AUTH_FILE="$HOME/.arc/agent/auth.json"
AUTH_BACKUP="$HOME/.arc/agent/auth.json.bak"

# Restore auth.json on exit (success or failure)
cleanup() {
    if [[ -f "$AUTH_BACKUP" ]]; then
        mv "$AUTH_BACKUP" "$AUTH_FILE"
        echo "Restored auth.json"
    fi
}
trap cleanup EXIT

# Move auth.json out of the way to prevent tests from using stored credentials
if [[ -f "$AUTH_FILE" ]]; then
    mv "$AUTH_FILE" "$AUTH_BACKUP"
    echo "Moved auth.json to backup"
fi

# Skip local LLM tests (ollama, etc.)
export ARC_NO_LOCAL_LLM=1

# Unset API keys to skip LLM-dependent tests
unset ANTHROPIC_API_KEY
unset ANTHROPIC_OAUTH_TOKEN
unset OPENAI_API_KEY
unset GEMINI_API_KEY
unset GROQ_API_KEY
unset CEREBRAS_API_KEY
unset XAI_API_KEY
unset OPENROUTER_API_KEY
unset MISTRAL_API_KEY
unset DEEPSEEK_API_KEY
unset GOOGLE_APPLICATION_CREDENTIALS
unset GOOGLE_CLOUD_PROJECT
unset GCLOUD_PROJECT
unset GOOGLE_CLOUD_LOCATION
unset AWS_PROFILE
unset AWS_ACCESS_KEY_ID
unset AWS_SECRET_ACCESS_KEY
unset AWS_SESSION_TOKEN
unset AWS_REGION
unset AWS_DEFAULT_REGION
unset AWS_BEARER_TOKEN_BEDROCK
unset AZURE_OPENAI_API_KEY
unset AZURE_OPENAI_BASE_URL

echo "Running tests without API keys..."

# Set PYTHONPATH to include all package sources
export PYTHONPATH="$SCRIPT_DIR/packages/ai/src:$SCRIPT_DIR/packages/agent/src:$SCRIPT_DIR/packages/tui/src:$SCRIPT_DIR/packages/coding-agent/src"

# Run pytest on all packages
pytest packages/ai/tests/ packages/agent/tests/ packages/tui/tests/ packages/coding-agent/tests/ "$@"
