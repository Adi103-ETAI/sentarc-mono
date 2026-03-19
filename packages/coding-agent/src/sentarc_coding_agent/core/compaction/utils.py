"""Token counting utilities for context compaction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


TOOL_RESULT_MAX_CHARS = 2000
SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a technical summarizer. Your job is to create structured context "
    "checkpoints from conversation histories. Be precise, comprehensive, and use "
    "exact file paths and code snippets when relevant."
)


def estimate_tokens(message: Any) -> int:
    """Rough token estimate for a message (1 token ≈ 4 chars)."""
    chars = 0
    role = message.get("role") if isinstance(message, dict) else getattr(message, "role", "")

    if role == "user":
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
        if isinstance(content, str):
            chars = len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    chars += len(block.get("text", ""))
    elif role == "assistant":
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type")
                    if btype == "text":
                        chars += len(block.get("text", ""))
                    elif btype == "thinking":
                        chars += len(block.get("thinking", ""))
                    elif btype == "toolCall":
                        args = block.get("arguments", {})
                        chars += len(str(args))
    elif role == "toolResult":
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    chars += len(block.get("text", "")[:TOOL_RESULT_MAX_CHARS])
    else:
        chars = 100  # Other message types

    return max(1, chars // 4)


def get_assistant_usage(message: Any) -> Optional[Dict[str, Any]]:
    """Get usage from an assistant message if available."""
    role = message.get("role") if isinstance(message, dict) else getattr(message, "role", "")
    if role != "assistant":
        return None
    usage = message.get("usage") if isinstance(message, dict) else getattr(message, "usage", None)
    stop_reason = message.get("stopReason") if isinstance(message, dict) else getattr(message, "stopReason", None)
    if stop_reason in ("aborted", "error") or not usage:
        return None
    return usage if isinstance(usage, dict) else vars(usage)


def calculate_context_tokens(usage: Dict[str, Any]) -> int:
    """Calculate total context tokens from usage."""
    if "totalTokens" in usage and usage["totalTokens"]:
        return usage["totalTokens"]
    return (
        usage.get("input", 0)
        + usage.get("output", 0)
        + usage.get("cacheRead", 0)
        + usage.get("cacheWrite", 0)
    )


def estimate_context_tokens(messages: List[Any]) -> Dict[str, Any]:
    """Estimate context tokens, using last assistant usage when available."""
    last_usage_info = None
    for i in range(len(messages) - 1, -1, -1):
        usage = get_assistant_usage(messages[i])
        if usage:
            last_usage_info = {"usage": usage, "index": i}
            break

    if last_usage_info is None:
        estimated = sum(estimate_tokens(m) for m in messages)
        return {
            "tokens": estimated,
            "usageTokens": 0,
            "trailingTokens": estimated,
            "lastUsageIndex": None,
        }

    usage_tokens = calculate_context_tokens(last_usage_info["usage"])
    trailing_tokens = sum(
        estimate_tokens(messages[i])
        for i in range(last_usage_info["index"] + 1, len(messages))
    )
    return {
        "tokens": usage_tokens + trailing_tokens,
        "usageTokens": usage_tokens,
        "trailingTokens": trailing_tokens,
        "lastUsageIndex": last_usage_info["index"],
    }


def create_file_ops() -> Dict[str, Set[str]]:
    return {"read": set(), "written": set(), "edited": set()}


def extract_file_ops_from_message(
    message: Any, file_ops: Dict[str, Set[str]]
) -> None:
    """Extract file operations from tool calls in an assistant message."""
    role = message.get("role") if isinstance(message, dict) else getattr(message, "role", "")
    if role != "assistant":
        return
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", [])
    if not isinstance(content, list):
        return
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "toolCall":
            continue
        args = block.get("arguments", {})
        name = block.get("name", "")
        path = args.get("path") if isinstance(args, dict) else None
        if not path:
            continue
        if name == "read":
            file_ops["read"].add(path)
        elif name == "write":
            file_ops["written"].add(path)
        elif name == "edit":
            file_ops["edited"].add(path)


def compute_file_lists(file_ops: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """Compute final file lists from file operations."""
    modified = file_ops["edited"] | file_ops["written"]
    read_only = sorted(f for f in file_ops["read"] if f not in modified)
    modified_files = sorted(modified)
    return {"readFiles": read_only, "modifiedFiles": modified_files}


def format_file_operations(read_files: List[str], modified_files: List[str]) -> str:
    """Format file operations as XML tags for summary."""
    sections = []
    if read_files:
        sections.append(f"<read-files>\n{chr(10).join(read_files)}\n</read-files>")
    if modified_files:
        sections.append(f"<modified-files>\n{chr(10).join(modified_files)}\n</modified-files>")
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


def truncate_for_summary(text: str, max_chars: int = TOOL_RESULT_MAX_CHARS) -> str:
    """Truncate text for summarization."""
    if len(text) <= max_chars:
        return text
    truncated_chars = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[... {truncated_chars} more characters truncated]"


def serialize_conversation(messages: List[Any]) -> str:
    """Serialize conversation messages to text for summarization."""
    lines = []
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
        if role == "user":
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            if isinstance(content, list):
                texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                text = "\n".join(texts)
            else:
                text = str(content)
            lines.append(f"User: {text}")
        elif role == "assistant":
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", [])
            parts = []
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        parts.append(block.get("text", ""))
                    elif btype == "toolCall":
                        name = block.get("name", "")
                        args = block.get("arguments", {})
                        parts.append(f"[Tool: {name}({args})]")
            lines.append(f"Assistant: {' '.join(parts)}")
        elif role == "toolResult":
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", [])
            if isinstance(content, list):
                texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                text = truncate_for_summary("\n".join(texts))
            else:
                text = truncate_for_summary(str(content))
            lines.append(f"Tool Result: {text}")
    return "\n\n".join(lines)
