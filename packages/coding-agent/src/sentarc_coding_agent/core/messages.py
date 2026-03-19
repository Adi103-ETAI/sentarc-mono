"""Custom agent message types for bash execution, compaction, and branch summaries."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

COMPACTION_SUMMARY_PREFIX = (
    "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
)
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"

BRANCH_SUMMARY_PREFIX = (
    "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n"
)
BRANCH_SUMMARY_SUFFIX = "</summary>"


def bash_execution_to_text(msg: Dict[str, Any]) -> str:
    """Convert a BashExecutionMessage to user message text for LLM context."""
    command = msg.get("command", "")
    output = msg.get("output", "")
    cancelled = msg.get("cancelled", False)
    exit_code = msg.get("exitCode")
    truncated = msg.get("truncated", False)
    full_output_path = msg.get("fullOutputPath")

    text = f"Ran `{command}`\n"
    if output:
        text += f"```\n{output}\n```"
    else:
        text += "(no output)"

    if cancelled:
        text += "\n\n(command cancelled)"
    elif exit_code is not None and exit_code != 0:
        text += f"\n\nCommand exited with code {exit_code}"

    if truncated and full_output_path:
        text += f"\n\n[Output truncated. Full output: {full_output_path}]"

    return text


def create_branch_summary_message(summary: str, from_id: str, timestamp: str) -> Dict[str, Any]:
    from datetime import datetime
    try:
        ts = int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        ts = 0
    return {
        "role": "branchSummary",
        "summary": summary,
        "fromId": from_id,
        "timestamp": ts,
    }


def create_compaction_summary_message(
    summary: str, tokens_before: int, timestamp: str
) -> Dict[str, Any]:
    from datetime import datetime
    try:
        ts = int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        ts = 0
    return {
        "role": "compactionSummary",
        "summary": summary,
        "tokensBefore": tokens_before,
        "timestamp": ts,
    }


def create_custom_message(
    custom_type: str,
    content: Union[str, List[Dict[str, Any]]],
    display: bool,
    details: Any,
    timestamp: str,
) -> Dict[str, Any]:
    from datetime import datetime
    try:
        ts = int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        ts = 0
    return {
        "role": "custom",
        "customType": custom_type,
        "content": content,
        "display": display,
        "details": details,
        "timestamp": ts,
    }


def convert_to_llm(messages: List[Any]) -> List[Dict[str, Any]]:
    """
    Transform AgentMessages (including custom types) to LLM-compatible Messages.
    """
    result = []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)

        if role == "bashExecution":
            exclude = m.get("excludeFromContext", False) if isinstance(m, dict) else False
            if exclude:
                continue
            ts = m.get("timestamp", 0) if isinstance(m, dict) else getattr(m, "timestamp", 0)
            result.append({
                "role": "user",
                "content": [{"type": "text", "text": bash_execution_to_text(m)}],
                "timestamp": ts,
            })
        elif role == "custom":
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            ts = m.get("timestamp", 0) if isinstance(m, dict) else getattr(m, "timestamp", 0)
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            result.append({"role": "user", "content": content, "timestamp": ts})
        elif role == "branchSummary":
            summary = m.get("summary", "") if isinstance(m, dict) else getattr(m, "summary", "")
            ts = m.get("timestamp", 0) if isinstance(m, dict) else getattr(m, "timestamp", 0)
            result.append({
                "role": "user",
                "content": [{"type": "text", "text": BRANCH_SUMMARY_PREFIX + summary + BRANCH_SUMMARY_SUFFIX}],
                "timestamp": ts,
            })
        elif role == "compactionSummary":
            summary = m.get("summary", "") if isinstance(m, dict) else getattr(m, "summary", "")
            ts = m.get("timestamp", 0) if isinstance(m, dict) else getattr(m, "timestamp", 0)
            result.append({
                "role": "user",
                "content": [{"type": "text", "text": COMPACTION_SUMMARY_PREFIX + summary + COMPACTION_SUMMARY_SUFFIX}],
                "timestamp": ts,
            })
        elif role in ("user", "assistant", "toolResult"):
            result.append(m if isinstance(m, dict) else vars(m))
        # Other roles are skipped

    return result
