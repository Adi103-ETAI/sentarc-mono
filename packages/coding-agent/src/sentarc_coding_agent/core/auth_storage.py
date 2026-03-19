"""OAuth token storage in ~/.arc/agent/auth.json."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from sentarc_coding_agent.config import get_auth_path


def load_auth() -> Dict[str, Any]:
    """Load auth tokens from file."""
    path = get_auth_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_auth(data: Dict[str, Any]) -> None:
    """Save auth tokens to file."""
    path = get_auth_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_token(provider: str) -> Optional[str]:
    """Get OAuth token for a provider."""
    auth = load_auth()
    return auth.get(provider, {}).get("token")


def set_token(provider: str, token: str) -> None:
    """Store OAuth token for a provider."""
    auth = load_auth()
    if provider not in auth:
        auth[provider] = {}
    auth[provider]["token"] = token
    save_auth(auth)


def clear_token(provider: str) -> None:
    """Clear OAuth token for a provider."""
    auth = load_auth()
    if provider in auth:
        del auth[provider]
        save_auth(auth)
