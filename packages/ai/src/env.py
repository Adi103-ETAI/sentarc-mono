"""
Environment variable management for API keys.
"""
import os
from pathlib import Path
from typing import Optional

def _get_vertex_adc_path() -> Optional[Path]:
    """Get path to Application Default Credentials JSON."""
    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        return Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    
    # Default location: ~/.config/gcloud/application_default_credentials.json
    home = Path.home()
    return home / ".config" / "gcloud" / "application_default_credentials.json"

def has_vertex_adc_credentials() -> bool:
    """Check if Vertex AI credentials exist."""
    path = _get_vertex_adc_path()
    return path.exists() if path else False

def get_env_api_key(provider: str) -> Optional[str]:
    """Get API key for provider from environment variables."""
    
    # Github Copilot
    if provider == "github-copilot":
        return os.environ.get("COPILOT_GITHUB_TOKEN") or \
               os.environ.get("GH_TOKEN") or \
               os.environ.get("GITHUB_TOKEN")

    # Anthropic
    if provider == "anthropic":
        return os.environ.get("ANTHROPIC_OAUTH_TOKEN") or \
               os.environ.get("ANTHROPIC_API_KEY")

    # Vertex AI (ADC check)
    if provider == "google-vertex":
        has_creds = has_vertex_adc_credentials()
        has_project = bool(os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT"))
        has_location = bool(os.environ.get("GOOGLE_CLOUD_LOCATION"))
        
        if has_creds and has_project and has_location:
            return "<authenticated>"
        return None

    # Amazon Bedrock
    if provider == "amazon-bedrock":
        if (os.environ.get("AWS_PROFILE") or
            (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")) or
            os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or
            os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI") or
            os.environ.get("AWS_CONTAINER_CREDENTIALS_FULL_URI") or
            os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE")):
            return "<authenticated>"
        return None

    # Standard mapping
    ENV_MAP = {
        "openai":                 "OPENAI_API_KEY",
        "azure-openai-responses": "AZURE_OPENAI_API_KEY",
        "google":                 "GEMINI_API_KEY",
        "groq":                   "GROQ_API_KEY",
        "cerebras":               "CEREBRAS_API_KEY",
        "xai":                    "XAI_API_KEY",
        "openrouter":             "OPENROUTER_API_KEY",
        "vercel-ai-gateway":      "AI_GATEWAY_API_KEY",
        "zai":                    "ZAI_API_KEY",
        "mistral":                "MISTRAL_API_KEY",
        "minimax":                "MINIMAX_API_KEY",
        "minimax-cn":             "MINIMAX_CN_API_KEY",
        "huggingface":            "HF_TOKEN",
        "opencode":               "OPENCODE_API_KEY",
        "kimi-coding":            "KIMI_API_KEY",
        "ollama":                 None, # No key usually
    }
    
    env_var = ENV_MAP.get(provider)
    return os.environ.get(env_var) if env_var else None
