"""Send desktop notifications when the assistant responds."""
from __future__ import annotations

import subprocess
import shutil

EXTENSION_NAME = "notify"


def _notify(title: str, body: str) -> None:
    if shutil.which("notify-send"):
        subprocess.run(["notify-send", title, body], check=False)
    else:
        print(f"[notification] {title}: {body}")


def on_agent_end(ctx, messages=None):
    if not messages:
        return
    text_blocks = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    text_blocks.append(block.get("text", ""))
    if not text_blocks:
        return
    preview = text_blocks[-1][:140]
    _notify("arc reply", preview)
