"""Shell config utilities."""

from __future__ import annotations

import os
import shutil
from typing import Dict, Tuple


def get_shell() -> str:
    """Get the user's preferred shell."""
    shell = os.environ.get("SHELL", "")
    if shell and os.path.isfile(shell):
        return shell
    for s in ("bash", "sh"):
        found = shutil.which(s)
        if found:
            return found
    return "/bin/sh"


def get_shell_env() -> Dict[str, str]:
    """Get environment for shell execution."""
    env = os.environ.copy()
    env.pop("TERM_PROGRAM", None)
    return env
