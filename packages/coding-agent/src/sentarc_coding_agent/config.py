"""Configuration constants and path helpers for sentarc-coding-agent."""

import os
import sys
from pathlib import Path

APP_NAME: str = "arc"
CONFIG_DIR_NAME: str = ".arc"
VERSION: str = "0.1.0"

ENV_AGENT_DIR: str = "ARC_CODING_AGENT_DIR"

DEFAULT_SHARE_VIEWER_URL = "https://sentarc.dev/session/"


def get_share_viewer_url(gist_id: str) -> str:
    """Return the share viewer URL for a gist ID."""
    base_url = os.environ.get("ARC_SHARE_VIEWER_URL", DEFAULT_SHARE_VIEWER_URL)
    return f"{base_url}#{gist_id}"


def get_package_dir() -> str:
    """Return the package root directory."""
    env_dir = os.environ.get("ARC_PACKAGE_DIR")
    if env_dir:
        if env_dir == "~":
            return str(Path.home())
        if env_dir.startswith("~/"):
            return str(Path.home() / env_dir[2:])
        return env_dir
    # Walk up from this file until we find pyproject.toml
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists():
            return str(d)
        d = d.parent
    return str(Path(__file__).resolve().parent)


def get_readme_path() -> str:
    return str(Path(get_package_dir()) / "README.md")


def get_docs_path() -> str:
    return str(Path(get_package_dir()) / "docs")


def get_examples_path() -> str:
    return str(Path(get_package_dir()) / "examples")


def get_changelog_path() -> str:
    return str(Path(get_package_dir()) / "CHANGELOG.md")


# =============================================================================

def get_agent_dir() -> str:
    """Return the agent config directory (~/.arc/agent/)."""
    env_dir = os.environ.get(ENV_AGENT_DIR)
    if env_dir:
        if env_dir == "~":
            return str(Path.home())
        if env_dir.startswith("~/"):
            return str(Path.home() / env_dir[2:])
        return env_dir
    return str(Path.home() / CONFIG_DIR_NAME / "agent")


def get_custom_themes_dir() -> str:
    return str(Path(get_agent_dir()) / "themes")


def get_models_path() -> str:
    return str(Path(get_agent_dir()) / "models.json")


def get_auth_path() -> str:
    return str(Path(get_agent_dir()) / "auth.json")


def get_settings_path() -> str:
    return str(Path(get_agent_dir()) / "settings.json")


def get_tools_dir() -> str:
    return str(Path(get_agent_dir()) / "tools")


def get_bin_dir() -> str:
    return str(Path(get_agent_dir()) / "bin")


def get_prompts_dir() -> str:
    return str(Path(get_agent_dir()) / "prompts")


def get_sessions_dir() -> str:
    return str(Path(get_agent_dir()) / "sessions")


def get_debug_log_path() -> str:
    return str(Path(get_agent_dir()) / f"{APP_NAME}-debug.log")
