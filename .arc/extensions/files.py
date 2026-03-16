"""files — /files command lists files read/written/edited in the current session."""

from typing import Any, Dict, Set

EXTENSION_NAME = "files"


async def _files_handler(args: str, ctx: Any) -> str:
    """List files the model has read/written/edited in the active session."""
    session_manager = ctx.session_manager
    if not session_manager:
        return "No session manager available."

    entries = session_manager.get_entries()

    # Collect tool calls from assistant messages
    tool_calls: Dict[str, Dict[str, Any]] = {}  # id -> {path, name, timestamp}
    for entry in entries:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "toolCall":
                continue
            name = block.get("name", "")
            if name in ("read", "write", "edit"):
                arguments = block.get("arguments", {})
                path = arguments.get("path", "") if isinstance(arguments, dict) else ""
                if path:
                    tool_calls[block.get("id", "")] = {
                        "path": path,
                        "name": name,
                        "timestamp": msg.get("timestamp", ""),
                    }

    # Match tool results
    file_map: Dict[str, Dict[str, Any]] = {}  # path -> {operations, timestamp}
    for entry in entries:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        if msg.get("role") != "toolResult":
            continue
        tc = tool_calls.get(msg.get("toolCallId", ""))
        if not tc:
            continue

        path = tc["path"]
        name = tc["name"]
        timestamp = msg.get("timestamp", "")

        if path in file_map:
            file_map[path]["operations"].add(name)
            if timestamp > file_map[path]["timestamp"]:
                file_map[path]["timestamp"] = timestamp
        else:
            file_map[path] = {"operations": {name}, "timestamp": timestamp}

    if not file_map:
        return "No files read/written/edited in this session."

    # Sort newest first
    sorted_files = sorted(file_map.items(), key=lambda x: x[1]["timestamp"], reverse=True)

    lines = [f"Session files ({len(sorted_files)}):"]
    for path, info in sorted_files:
        ops = []
        if "read" in info["operations"]:
            ops.append("R")
        if "write" in info["operations"]:
            ops.append("W")
        if "edit" in info["operations"]:
            ops.append("E")
        lines.append(f"  {''.join(ops)} {path}")

    return "\n".join(lines)


COMMANDS = [
    {
        "name": "files",
        "description": "Show files read/written/edited in this session",
        "handler": _files_handler,
    },
]
