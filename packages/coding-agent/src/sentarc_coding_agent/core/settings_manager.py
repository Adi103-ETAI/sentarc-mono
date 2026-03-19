"""Read/write settings from ~/.arc/settings.json."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from sentarc_coding_agent.config import get_settings_path


@dataclass
class Settings:
    provider: str = "google"
    model: str = "gemini-2.5-flash"
    thinking: str = "off"
    quiet_startup: bool = False
    tools: List[str] = field(default_factory=lambda: ["read", "bash", "edit", "write"])
    extra: Dict[str, Any] = field(default_factory=dict)


def load_settings() -> Settings:
    """Load settings from file, returning defaults if not found."""
    path = get_settings_path()
    if not os.path.exists(path):
        return Settings()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        s = Settings()
        s.provider = data.get("provider", s.provider)
        s.model = data.get("model", s.model)
        s.thinking = data.get("thinking", s.thinking)
        s.quiet_startup = data.get("quietStartup", s.quiet_startup)
        s.tools = data.get("tools", s.tools)
        return s
    except Exception:
        return Settings()


def save_settings(settings: Settings) -> None:
    """Save settings to file."""
    path = get_settings_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "provider": settings.provider,
        "model": settings.model,
        "thinking": settings.thinking,
        "quietStartup": settings.quiet_startup,
        "tools": settings.tools,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
