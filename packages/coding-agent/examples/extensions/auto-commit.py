"""Optionally commit work on session end.

Enable by setting ARC_AUTO_COMMIT=1. Commits all tracked changes in the current repo.
"""
from __future__ import annotations

import os
import subprocess
from datetime import datetime

EXTENSION_NAME = "auto-commit"


def _in_repo(cwd: str) -> bool:
    try:
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def _has_changes(cwd: str) -> bool:
    try:
        result = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, check=True, capture_output=True, text=True)
        return bool(result.stdout.strip())
    except Exception:
        return False


def on_session_end(ctx):
    if os.environ.get("ARC_AUTO_COMMIT") != "1":
        return
    if not _in_repo(ctx.cwd) or not _has_changes(ctx.cwd):
        return
    subprocess.run(["git", "add", "-A"], cwd=ctx.cwd, check=False)
    msg = datetime.now().strftime("arc auto-commit %Y-%m-%d %H:%M:%S")
    subprocess.run(["git", "commit", "-m", msg], cwd=ctx.cwd, check=False)
