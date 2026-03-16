"""Shell configuration for bash tool execution."""

from __future__ import annotations

import os
import shutil
from typing import Dict, Optional, Tuple


def get_shell_config() -> Tuple[str, list]:
    """Get the shell and args for subprocess execution."""
    shell = os.environ.get("SHELL", "/bin/bash")
    if not os.path.isfile(shell):
        shell = shutil.which("bash") or "/bin/sh"
    return shell, ["-c"]


def get_shell_env() -> Dict[str, str]:
    """Get a clean environment for shell execution."""
    env = os.environ.copy()
    # Remove any terminal-specific variables that could cause issues
    env.pop("TERM_PROGRAM", None)
    return env
