#!/usr/bin/env bash
# Run arc from sources (must be run from repo root)
# Usage: ./arc-test.sh [arc options]
#        ./arc-test.sh --no-env [arc options]  # Run without API keys
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure we're at repo root
if [[ ! -d "$SCRIPT_DIR/packages/coding-agent" ]]; then
    echo "Error: Must be run from sentarc-mono repo root" >&2
    exit 1
fi

# Check for --no-env flag to disable API keys
NO_ENV=false
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--no-env" ]]; then
        NO_ENV=true
    else
        ARGS+=("$arg")
    fi
done

if [[ "$NO_ENV" == "true" ]]; then
    echo "Running without API keys..."
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
fi

# Run arc from sources by setting PYTHONPATH
PYTHONPATH="$SCRIPT_DIR/packages/ai/src:$SCRIPT_DIR/packages/agent/src:$SCRIPT_DIR/packages/tui/src:$SCRIPT_DIR/packages/coding-agent/src" \
    python -m sentarc_coding_agent.cli ${ARGS[@]+"${ARGS[@]}"}
