"""prompt_url_widget — Detects GitHub PR/issue URLs in prompts and displays metadata."""

import json
import re
import subprocess
from typing import Any, Dict, Optional, Tuple

EXTENSION_NAME = "prompt-url-widget"

PR_PROMPT_PATTERN = re.compile(
    r"^\s*You are given one or more GitHub PR URLs:\s*(\S+)", re.IGNORECASE | re.MULTILINE
)
ISSUE_PROMPT_PATTERN = re.compile(
    r"^\s*Analyze GitHub issue\(s\):\s*(\S+)", re.IGNORECASE | re.MULTILINE
)


def _extract_prompt_match(prompt: str) -> Optional[Tuple[str, str]]:
    """Returns (kind, url) or None."""
    m = PR_PROMPT_PATTERN.search(prompt)
    if m:
        return ("pr", m.group(1).strip())
    m = ISSUE_PROMPT_PATTERN.search(prompt)
    if m:
        return ("issue", m.group(1).strip())
    return None


def _fetch_gh_metadata(kind: str, url: str, cwd: str) -> Optional[Dict[str, Any]]:
    """Fetch title and author via gh CLI."""
    if kind == "pr":
        cmd = ["gh", "pr", "view", url, "--json", "title,author"]
    else:
        cmd = ["gh", "issue", "view", url, "--json", "title,author"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=10)
        if result.returncode != 0 or not result.stdout:
            return None
        return json.loads(result.stdout)
    except Exception:
        return None


def _format_author(author: Optional[Dict[str, Any]]) -> Optional[str]:
    if not author:
        return None
    name = (author.get("name") or "").strip()
    login = (author.get("login") or "").strip()
    if name and login:
        return f"{name} (@{login})"
    if login:
        return f"@{login}"
    if name:
        return name
    return None


def _get_user_text(content: Any) -> str:
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return ""


def on_before_agent_start(ctx: Any, **kwargs: Any) -> None:
    """Check user prompt for PR/issue URLs and log metadata."""
    prompt = kwargs.get("prompt", "")
    match = _extract_prompt_match(prompt)
    if not match:
        return

    kind, url = match
    label = "PR" if kind == "pr" else "Issue"

    meta = _fetch_gh_metadata(kind, url, ctx.cwd)
    title = (meta.get("title", "") or "").strip() if meta else ""
    author = _format_author(meta.get("author")) if meta else None

    parts = [f"{label}: {title or url}"]
    if author:
        parts.append(f"  Author: {author}")
    parts.append(f"  URL: {url}")

    if ctx.notify:
        ctx.notify("\n".join(parts), "info")


def on_session_start(ctx: Any, **kwargs: Any) -> None:
    """Rebuild URL widget when a session is loaded."""
    session_manager = ctx.session_manager
    if not session_manager:
        return

    entries = session_manager.get_entries()
    # Search backwards for a user message with a PR/issue URL
    for entry in reversed(entries):
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        if msg.get("role") != "user":
            continue
        text = _get_user_text(msg.get("content"))
        match = _extract_prompt_match(text)
        if match:
            kind, url = match
            label = "PR" if kind == "pr" else "Issue"
            if ctx.notify:
                ctx.notify(f"{label}: {url}", "info")
            return
