"""Shows simple stats in the interactive footer after each run."""
from __future__ import annotations

EXTENSION_NAME = "custom-footer"


def on_agent_end(ctx, messages=None):
    if not ctx.has_ui or not ctx.notify:
        return
    entry_count = len(ctx.session_manager.get_entries()) if ctx.session_manager else 0
    ctx.notify(f"Session entries: {entry_count}", "info")
