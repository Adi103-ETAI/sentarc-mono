"""Git utilities."""

from __future__ import annotations

import os
import subprocess
from typing import Optional


def get_git_root(cwd: Optional[str] = None) -> Optional[str]:
    """Get the git repository root from a directory."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_current_branch(cwd: Optional[str] = None) -> Optional[str]:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def is_git_repo(cwd: Optional[str] = None) -> bool:
    """Check if the current directory is a git repository."""
    return get_git_root(cwd) is not None
