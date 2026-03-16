"""tps — Tracks tokens-per-second stats across agent turns."""

import time
from typing import Any, Dict, List, Optional

EXTENSION_NAME = "tps"

_agent_start_time: Optional[float] = None


def on_agent_start(ctx: Any, **kwargs: Any) -> None:
    """Record the timestamp when the agent starts processing."""
    global _agent_start_time
    _agent_start_time = time.monotonic()


def on_agent_end(ctx: Any, **kwargs: Any) -> None:
    """Calculate and display TPS stats when the agent finishes."""
    global _agent_start_time
    if _agent_start_time is None:
        return

    elapsed = time.monotonic() - _agent_start_time
    _agent_start_time = None
    if elapsed <= 0:
        return

    messages = kwargs.get("messages", [])
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_write = 0
    total_tokens = 0

    for msg in messages:
        if isinstance(msg, dict):
            if msg.get("role") != "assistant":
                continue
            usage = msg.get("usage", {})
        else:
            if getattr(msg, "role", "") != "assistant":
                continue
            usage = getattr(msg, "usage", {})
            if not isinstance(usage, dict):
                usage = vars(usage) if hasattr(usage, "__dict__") else {}

        input_tokens += usage.get("input", 0) or 0
        output_tokens += usage.get("output", 0) or 0
        cache_read += usage.get("cacheRead", 0) or usage.get("cache_read", 0) or 0
        cache_write += usage.get("cacheWrite", 0) or usage.get("cache_write", 0) or 0
        total_tokens += usage.get("totalTokens", 0) or usage.get("total_tokens", 0) or 0

    if output_tokens <= 0:
        return

    tps = output_tokens / elapsed
    msg = (
        f"TPS {tps:.1f} tok/s | "
        f"out {output_tokens:,} | in {input_tokens:,} | "
        f"cache r/w {cache_read:,}/{cache_write:,} | "
        f"total {total_tokens:,} | {elapsed:.1f}s"
    )

    if ctx.notify:
        ctx.notify(msg, "info")
