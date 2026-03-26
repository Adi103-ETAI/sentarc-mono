"""Read/write settings from global and project settings JSON files."""

from __future__ import annotations

import json
import os
from pathlib import Path
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
    bash_security_profile: str = "standard"
    bash_block_patterns: List[str] = field(default_factory=list)
    event_log_enabled: bool = False
    event_log_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _read_settings_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _get_project_settings_path(cwd: Optional[str]) -> str:
    base = Path(cwd or os.getcwd())
    return str(base / ".arc" / "settings.json")


def _apply_settings_overrides(settings: Settings, data: Dict[str, Any]) -> None:
    if not data:
        return

    if "provider" in data:
        settings.provider = data["provider"]
    if "model" in data:
        settings.model = data["model"]
    if "thinking" in data:
        settings.thinking = data["thinking"]
    if "quietStartup" in data:
        settings.quiet_startup = data["quietStartup"]
    elif "quiet_startup" in data:
        settings.quiet_startup = data["quiet_startup"]
    if "tools" in data:
        settings.tools = data["tools"]
    if "bashSecurityProfile" in data:
        settings.bash_security_profile = data["bashSecurityProfile"]
    if "bashBlockPatterns" in data:
        settings.bash_block_patterns = data["bashBlockPatterns"]
    if "eventLogEnabled" in data:
        settings.event_log_enabled = data["eventLogEnabled"]
    if "eventLogPath" in data:
        settings.event_log_path = data["eventLogPath"]


def load_settings(cwd: Optional[str] = None) -> Settings:
    """Load settings by merging global then project overrides."""
    settings = Settings()
    global_data = _read_settings_json(get_settings_path())
    project_data = _read_settings_json(_get_project_settings_path(cwd))

    _apply_settings_overrides(settings, global_data)
    _apply_settings_overrides(settings, project_data)
    return settings


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
        "bashSecurityProfile": settings.bash_security_profile,
        "bashBlockPatterns": settings.bash_block_patterns,
        "eventLogEnabled": settings.event_log_enabled,
        "eventLogPath": settings.event_log_path,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
