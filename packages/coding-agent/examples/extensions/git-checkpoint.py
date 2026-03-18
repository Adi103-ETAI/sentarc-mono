"""Optionally stash work after each agent run.

Enable by setting ARC_CHECKPOINT=1 when loading the extension.
"""
from __future__ import annotations

import os
import subprocess
from datetime import datetime

EXTENSION_NAME = "git-checkpoint"


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


def on_agent_end(ctx, messages=None):
    if os.environ.get("ARC_CHECKPOINT") != "1":
        return
    if not _in_repo(ctx.cwd):
        return
    if not _has_changes(ctx.cwd):
        return
    label = datetime.now().strftime("arc-checkpoint-%Y%m%d-%H%M%S")
    subprocess.run(["git", "stash", "push", "-m", label], cwd=ctx.cwd, check=False)
