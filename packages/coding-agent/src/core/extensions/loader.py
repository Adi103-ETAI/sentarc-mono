"""Discover and load Python extension modules."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentarc_coding_agent.core.extensions.types import (
    Extension,
    ExtensionCommand,
    ExtensionFlag,
)


def load_extension_from_path(path: str) -> Optional[Extension]:
    """Load a Python extension from a file path."""
    if not os.path.isfile(path):
        return None

    try:
        spec = importlib.util.spec_from_file_location("_ext", path)
        if not spec or not spec.loader:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        name = getattr(module, "EXTENSION_NAME", Path(path).stem)
        flags_raw: Dict[str, Any] = getattr(module, "EXTENSION_FLAGS", {})
        flags = {
            k: ExtensionFlag(
                type=v.get("type", "boolean"),
                description=v.get("description", ""),
            )
            for k, v in flags_raw.items()
        }

        # Load registered commands
        commands_raw: List[Dict[str, Any]] = getattr(module, "COMMANDS", [])
        commands = [
            ExtensionCommand(
                name=c["name"],
                description=c.get("description", ""),
                handler=c["handler"],
            )
            for c in commands_raw
            if "name" in c and "handler" in c
        ]

        return Extension(
            name=name,
            path=path,
            flags=flags,
            # Lifecycle hooks
            on_start=getattr(module, "on_start", None),
            on_message=getattr(module, "on_message", None),
            on_tool_call=getattr(module, "on_tool_call", None),
            on_session_end=getattr(module, "on_session_end", None),
            # Event hooks
            on_agent_start=getattr(module, "on_agent_start", None),
            on_agent_end=getattr(module, "on_agent_end", None),
            on_before_agent_start=getattr(module, "on_before_agent_start", None),
            on_session_switch=getattr(module, "on_session_switch", None),
            on_session_start=getattr(module, "on_session_start", None),
            commands=commands,
        )
    except Exception as e:
        print(f"Warning: Failed to load extension {path}: {e}")
        return None


def _scan_dir(dir_path: str) -> List[Extension]:
    """Load all .py extensions from a directory."""
    extensions: List[Extension] = []
    if not os.path.isdir(dir_path):
        return extensions
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith(".py") and not fname.startswith("_"):
            ext = load_extension_from_path(os.path.join(dir_path, fname))
            if ext:
                extensions.append(ext)
    return extensions


def discover_extensions(
    extension_paths: Optional[List[str]] = None,
    agent_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    no_extensions: bool = False,
) -> List[Extension]:
    """Discover and load extensions from configured paths.

    Scans (in order):
    1. User-global: ~/.arc/agent/tools/*.py
    2. Project-level: <cwd>/.arc/extensions/*.py
    3. Explicit paths from extension_paths
    """
    from sentarc_coding_agent.config import CONFIG_DIR_NAME

    extensions: List[Extension] = []

    if no_extensions:
        return extensions

    # 1. User-global extensions
    if agent_dir:
        extensions.extend(_scan_dir(os.path.join(agent_dir, "tools")))

    # 2. Project-level extensions
    if project_dir:
        extensions.extend(
            _scan_dir(os.path.join(project_dir, CONFIG_DIR_NAME, "extensions"))
        )

    # 3. Explicit paths
    for path in extension_paths or []:
        expanded = os.path.expanduser(path)
        if not os.path.isabs(expanded):
            expanded = os.path.join(os.getcwd(), expanded)
        ext = load_extension_from_path(expanded)
        if ext:
            extensions.append(ext)

    return extensions
