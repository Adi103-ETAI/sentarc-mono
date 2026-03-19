"""Session manager — persists conversation history as JSONL files."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sentarc_coding_agent.config import get_sessions_dir
from sentarc_coding_agent.core.messages import (
    create_branch_summary_message,
    create_compaction_summary_message,
    create_custom_message,
)

CURRENT_SESSION_VERSION = 3


def _generate_id(existing: set) -> str:
    for _ in range(100):
        id_ = str(uuid.uuid4())[:8]
        if id_ not in existing:
            return id_
    return str(uuid.uuid4())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _encode_cwd(cwd: str) -> str:
    """Encode cwd into a safe directory name (same logic as TS)."""
    return cwd.replace("/", "--").replace("\\", "--").replace(":", "")


def get_default_session_dir(cwd: str) -> str:
    """Get default session directory for a cwd."""
    sessions_dir = get_sessions_dir()
    encoded = _encode_cwd(cwd)
    return str(Path(sessions_dir) / encoded)


def parse_session_entries(content: str) -> List[Dict[str, Any]]:
    """Parse JSONL content into list of entry dicts."""
    entries = []
    errors = []
    for line_num, line in enumerate(content.strip().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as e:
            errors.append(f"Line {line_num}: {e.msg}")
    if errors:
        import sys
        print(f"Warning: Failed to parse {len(errors)} session entries:", file=sys.stderr)
        for err in errors[:5]:
            print(f"  {err}", file=sys.stderr)
    return entries


def _migrate_v1_to_v2(entries: List[Dict[str, Any]]) -> None:
    ids: set = set()
    prev_id: Optional[str] = None
    for entry in entries:
        if entry.get("type") == "session":
            entry["version"] = 2
            continue
        entry["id"] = _generate_id(ids)
        ids.add(entry["id"])
        entry["parentId"] = prev_id
        prev_id = entry["id"]
        # Convert firstKeptEntryIndex to firstKeptEntryId
        if entry.get("type") == "compaction" and "firstKeptEntryIndex" in entry:
            idx = entry["firstKeptEntryIndex"]
            if isinstance(idx, int) and 0 <= idx < len(entries):
                target = entries[idx]
                if target.get("type") != "session":
                    entry["firstKeptEntryId"] = target.get("id", "")
            del entry["firstKeptEntryIndex"]


def _migrate_v2_to_v3(entries: List[Dict[str, Any]]) -> None:
    for entry in entries:
        if entry.get("type") == "session":
            entry["version"] = 3
            continue
        if entry.get("type") == "message":
            msg = entry.get("message", {})
            if isinstance(msg, dict) and msg.get("role") == "hookMessage":
                msg["role"] = "custom"


def migrate_session_entries(entries: List[Dict[str, Any]]) -> bool:
    """Run all migrations. Returns True if any migration was applied."""
    header = next((e for e in entries if e.get("type") == "session"), None)
    version = header.get("version", 1) if header else 1
    if version >= CURRENT_SESSION_VERSION:
        return False
    if version < 2:
        _migrate_v1_to_v2(entries)
    if version < 3:
        _migrate_v2_to_v3(entries)
    return True


def load_entries_from_file(path: str) -> List[Dict[str, Any]]:
    """Load and migrate entries from a JSONL file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return []
    entries = parse_session_entries(content)
    migrate_session_entries(entries)
    return entries


def get_latest_compaction_entry(entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for e in reversed(entries):
        if e.get("type") == "compaction":
            return e
    return None


def find_most_recent_session(session_dir: str) -> Optional[str]:
    """Find the most recently modified .jsonl file in session_dir."""
    try:
        files = [
            f for f in os.listdir(session_dir) if f.endswith(".jsonl")
        ]
        if not files:
            return None
        files_with_mtime = [
            (os.path.getmtime(os.path.join(session_dir, f)), os.path.join(session_dir, f))
            for f in files
        ]
        files_with_mtime.sort(reverse=True)
        return files_with_mtime[0][1]
    except Exception:
        return None


def build_session_context(
    entries: List[Dict[str, Any]],
    leaf_id: Optional[str],
    by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build session context from tree of entries."""
    messages = []
    thinking_level = "off"
    model = None

    # Walk branch from leaf to root
    path: List[Dict[str, Any]] = []
    current_id = leaf_id
    while current_id:
        entry = by_id.get(current_id)
        if not entry:
            break
        path.insert(0, entry)
        current_id = entry.get("parentId")

    for entry in path:
        etype = entry.get("type")
        if etype == "message":
            msg = entry.get("message")
            if msg:
                messages.append(msg)
        elif etype == "custom_message":
            msg = create_custom_message(
                entry.get("customType", ""),
                entry.get("content", ""),
                entry.get("display", False),
                entry.get("details"),
                entry.get("timestamp", ""),
            )
            messages.append(msg)
        elif etype == "branch_summary":
            msg = create_branch_summary_message(
                entry.get("summary", ""),
                entry.get("fromId", ""),
                entry.get("timestamp", ""),
            )
            messages.append(msg)
        elif etype == "compaction":
            msg = create_compaction_summary_message(
                entry.get("summary", ""),
                entry.get("tokensBefore", 0),
                entry.get("timestamp", ""),
            )
            messages.append(msg)
        elif etype == "thinking_level_change":
            thinking_level = entry.get("thinkingLevel", "off")
        elif etype == "model_change":
            model = {"provider": entry.get("provider", ""), "modelId": entry.get("modelId", "")}

    return {"messages": messages, "thinkingLevel": thinking_level, "model": model}


class SessionManager:
    """
    Manages a JSONL session file with tree structure.
    Mirrors the TypeScript SessionManager class.
    """

    def __init__(
        self,
        cwd: str,
        session_dir: str,
        session_file: Optional[str],
        persist: bool,
    ) -> None:
        self.cwd = cwd
        self._session_dir = session_dir
        self._session_file = session_file
        self.persist = persist
        self._file_entries: List[Dict[str, Any]] = []
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._labels_by_id: Dict[str, str] = {}
        self._leaf_id: Optional[str] = None
        self._session_id: Optional[str] = None
        self._flushed: bool = False

        if session_file and os.path.exists(session_file):
            self._file_entries = load_entries_from_file(session_file)
            self._build_index()
        else:
            # New session
            self._session_id = str(uuid.uuid4())
            header: Dict[str, Any] = {
                "type": "session",
                "version": CURRENT_SESSION_VERSION,
                "id": self._session_id,
                "timestamp": _now_iso(),
                "cwd": cwd,
            }
            self._file_entries = [header]

    def _build_index(self) -> None:
        """Build by_id index and find leaf."""
        self._by_id = {}
        self._labels_by_id = {}

        for entry in self._file_entries:
            if entry.get("type") == "session":
                self._session_id = entry.get("id")
                continue
            eid = entry.get("id")
            if eid:
                self._by_id[eid] = entry
                if entry.get("type") == "label":
                    target = entry.get("targetId")
                    label = entry.get("label")
                    if target and label is not None:
                        self._labels_by_id[target] = label
        self._update_leaf_id()

    def _update_leaf_id(self) -> None:
        """Set leaf_id to the newest entry without children."""
        child_parents = {e.get("parentId") for e in self._by_id.values() if isinstance(e, dict)}
        candidates = [eid for eid in self._by_id.keys() if eid and eid not in child_parents]
        self._leaf_id = candidates[-1] if candidates else None

    def _append_entry(self, entry: Dict[str, Any]) -> None:
        self._file_entries.append(entry)
        eid = entry.get("id")
        if eid and entry.get("type") != "session":
            self._by_id[eid] = entry
        self._update_leaf_id()
        if self.persist:
            self._persist(entry)

    def _persist(self, entry: Dict[str, Any]) -> None:
        """Append entry to JSONL file."""
        if not self._session_dir:
            return

        # Only write file when we have an assistant message
        has_assistant = any(
            e.get("type") == "message" and
            isinstance(e.get("message"), dict) and
            e["message"].get("role") == "assistant"
            for e in self._file_entries
        )
        if not has_assistant:
            return

        if not self._flushed:
            # First time: write all entries
            os.makedirs(self._session_dir, exist_ok=True)
            if not self._session_file:
                ts = _now_iso().replace(":", "-").replace(".", "-")
                self._session_file = str(
                    Path(self._session_dir) / f"{ts}_{self._session_id}.jsonl"
                )
            with open(self._session_file, "w", encoding="utf-8") as f:
                for e in self._file_entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            self._flushed = True
        else:
            # Append single entry
            with open(self._session_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _rewrite_file(self) -> None:
        """Rewrite entire session file."""
        if not self._session_file:
            return
        with open(self._session_file, "w", encoding="utf-8") as f:
            for e in self._file_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def get_cwd(self) -> str:
        return self.cwd

    def get_session_dir(self) -> str:
        return self._session_dir

    def get_session_id(self) -> Optional[str]:
        return self._session_id

    def get_session_file(self) -> Optional[str]:
        return self._session_file

    def get_leaf_id(self) -> Optional[str]:
        return self._leaf_id

    def get_leaf_entry(self) -> Optional[Dict[str, Any]]:
        if self._leaf_id:
            return self._by_id.get(self._leaf_id)
        return None

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        return self._by_id.get(entry_id)

    def get_label(self, entry_id: str) -> Optional[str]:
        return self._labels_by_id.get(entry_id)

    def get_header(self) -> Optional[Dict[str, Any]]:
        return next((e for e in self._file_entries if e.get("type") == "session"), None)

    def get_entries(self) -> List[Dict[str, Any]]:
        return [e for e in self._file_entries if e.get("type") != "session"]

    def get_session_name(self) -> Optional[str]:
        for e in reversed(self._file_entries):
            if e.get("type") == "session_info" and e.get("name"):
                return e["name"]
        return None

    def get_branch(self, from_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Walk from entry to root, returning all entries in path order."""
        path: List[Dict[str, Any]] = []
        start_id = from_id or self._leaf_id
        current = self._by_id.get(start_id) if start_id else None
        while current:
            path.insert(0, current)
            parent_id = current.get("parentId")
            current = self._by_id.get(parent_id) if parent_id else None
        return path

    def build_session_context(self) -> Dict[str, Any]:
        """Build session context for LLM."""
        return build_session_context(self.get_entries(), self._leaf_id, self._by_id)

    def append_message(self, message: Dict[str, Any]) -> str:
        """Append a message entry."""
        entry: Dict[str, Any] = {
            "type": "message",
            "id": _generate_id(set(self._by_id.keys())),
            "parentId": self._leaf_id,
            "timestamp": _now_iso(),
            "message": message,
        }
        self._append_entry(entry)
        return entry["id"]

    def append_thinking_level_change(self, level: str) -> str:
        entry: Dict[str, Any] = {
            "type": "thinking_level_change",
            "id": _generate_id(set(self._by_id.keys())),
            "parentId": self._leaf_id,
            "timestamp": _now_iso(),
            "thinkingLevel": level,
        }
        self._append_entry(entry)
        return entry["id"]

    def append_model_change(self, provider: str, model_id: str) -> str:
        entry: Dict[str, Any] = {
            "type": "model_change",
            "id": _generate_id(set(self._by_id.keys())),
            "parentId": self._leaf_id,
            "timestamp": _now_iso(),
            "provider": provider,
            "modelId": model_id,
        }
        self._append_entry(entry)
        return entry["id"]

    def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: Any = None,
        from_hook: bool = False,
    ) -> str:
        entry: Dict[str, Any] = {
            "type": "compaction",
            "id": _generate_id(set(self._by_id.keys())),
            "parentId": self._leaf_id,
            "timestamp": _now_iso(),
            "summary": summary,
            "firstKeptEntryId": first_kept_entry_id,
            "tokensBefore": tokens_before,
        }
        if details is not None:
            entry["details"] = details
        if from_hook:
            entry["fromHook"] = True
        self._append_entry(entry)
        return entry["id"]

    def append_branch_summary(
        self,
        from_id: str,
        summary: str,
        details: Any = None,
        from_hook: bool = False,
    ) -> str:
        entry: Dict[str, Any] = {
            "type": "branch_summary",
            "id": _generate_id(set(self._by_id.keys())),
            "parentId": self._leaf_id,
            "timestamp": _now_iso(),
            "fromId": from_id,
            "summary": summary,
        }
        if details is not None:
            entry["details"] = details
        if from_hook:
            entry["fromHook"] = True
        self._append_entry(entry)
        return entry["id"]

    def append_custom_message(
        self,
        custom_type: str,
        content: Any,
        display: bool,
        details: Any = None,
    ) -> str:
        entry: Dict[str, Any] = {
            "type": "custom_message",
            "id": _generate_id(set(self._by_id.keys())),
            "parentId": self._leaf_id,
            "timestamp": _now_iso(),
            "customType": custom_type,
            "content": content,
            "display": display,
        }
        if details is not None:
            entry["details"] = details
        self._append_entry(entry)
        return entry["id"]

    def append_custom(self, custom_type: str, data: Any = None) -> str:
        entry: Dict[str, Any] = {
            "type": "custom",
            "id": _generate_id(set(self._by_id.keys())),
            "parentId": self._leaf_id,
            "timestamp": _now_iso(),
            "customType": custom_type,
        }
        if data is not None:
            entry["data"] = data
        self._append_entry(entry)
        return entry["id"]

    def set_label(self, target_id: str, label: Optional[str]) -> str:
        entry: Dict[str, Any] = {
            "type": "label",
            "id": _generate_id(set(self._by_id.keys())),
            "parentId": self._leaf_id,
            "timestamp": _now_iso(),
            "targetId": target_id,
            "label": label,
        }
        self._append_entry(entry)
        if label is not None:
            self._labels_by_id[target_id] = label
        else:
            self._labels_by_id.pop(target_id, None)
        return entry["id"]

    def branch(self, branch_from_id: str) -> None:
        """Start a new branch from an earlier entry."""
        if branch_from_id not in self._by_id:
            raise Exception(f"Entry {branch_from_id} not found")
        self._leaf_id = branch_from_id

    def reset_leaf(self) -> None:
        self._leaf_id = None

    def branch_with_summary(
        self,
        branch_from_id: Optional[str],
        summary: str,
        details: Any = None,
        from_hook: bool = False,
    ) -> str:
        if branch_from_id is not None and branch_from_id not in self._by_id:
            raise Exception(f"Entry {branch_from_id} not found")
        self._leaf_id = branch_from_id
        return self.append_branch_summary(branch_from_id or "root", summary, details, from_hook)

    @classmethod
    def create(cls, cwd: str, session_dir: Optional[str] = None) -> "SessionManager":
        dir_ = session_dir or get_default_session_dir(cwd)
        return cls(cwd, dir_, None, True)

    @classmethod
    def open(cls, path: str, session_dir: Optional[str] = None) -> "SessionManager":
        entries = load_entries_from_file(path)
        header = next((e for e in entries if e.get("type") == "session"), None)
        cwd = header.get("cwd", os.getcwd()) if header else os.getcwd()
        dir_ = session_dir or str(Path(path).parent)
        return cls(cwd, dir_, path, True)

    @classmethod
    def continue_recent(cls, cwd: str, session_dir: Optional[str] = None) -> "SessionManager":
        dir_ = session_dir or get_default_session_dir(cwd)
        most_recent = find_most_recent_session(dir_)
        if most_recent:
            return cls(cwd, dir_, most_recent, True)
        return cls(cwd, dir_, None, True)

    @classmethod
    def in_memory(cls, cwd: Optional[str] = None) -> "SessionManager":
        return cls(cwd or os.getcwd(), "", None, False)

    @classmethod
    async def list(
        cls,
        cwd: str,
        session_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all sessions for a directory."""
        dir_ = session_dir or get_default_session_dir(cwd)
        return await list_sessions_from_dir(dir_)

    @classmethod
    async def list_all(cls) -> List[Dict[str, Any]]:
        """List all sessions across all project directories."""
        sessions_dir = get_sessions_dir()
        if not os.path.exists(sessions_dir):
            return []
        sessions = []
        try:
            for entry in os.scandir(sessions_dir):
                if entry.is_dir():
                    sessions.extend(await list_sessions_from_dir(entry.path))
        except Exception:
            pass
        sessions.sort(key=lambda s: s.get("modified", ""), reverse=True)
        return sessions


async def list_sessions_from_dir(session_dir: str) -> List[Dict[str, Any]]:
    """List session info from a directory."""
    import asyncio
    sessions = []
    if not os.path.exists(session_dir):
        return sessions
    try:
        files = [f for f in os.listdir(session_dir) if f.endswith(".jsonl")]
    except Exception:
        return sessions

    for fname in files:
        path = os.path.join(session_dir, fname)
        info = await build_session_info(path)
        if info:
            sessions.append(info)

    sessions.sort(key=lambda s: s.get("modified", datetime.min), reverse=True)
    return sessions


async def build_session_info(path: str) -> Optional[Dict[str, Any]]:
    """Build session info dict from a JSONL file."""
    try:
        stat = os.stat(path)
        entries = load_entries_from_file(path)
        if not entries:
            return None

        header = next((e for e in entries if e.get("type") == "session"), None)
        if not header:
            return None

        session_entries = [e for e in entries if e.get("type") != "session"]
        message_entries = [e for e in session_entries if e.get("type") == "message"]
        message_count = len(message_entries)

        first_message = ""
        for e in session_entries:
            if e.get("type") == "message":
                msg = e.get("message", {})
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                first_message = c.get("text", "")[:200]
                                break
                    elif isinstance(content, str):
                        first_message = content[:200]
                    if first_message:
                        break

        # Get name from session_info entries
        name: Optional[str] = None
        for e in reversed(session_entries):
            if e.get("type") == "session_info" and e.get("name"):
                name = e["name"]
                break

        created = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

        return {
            "path": path,
            "id": header.get("id", ""),
            "cwd": header.get("cwd", ""),
            "name": name,
            "parentSessionPath": header.get("parentSession"),
            "created": created,
            "modified": modified,
            "messageCount": message_count,
            "firstMessage": first_message,
        }
    except Exception:
        return None
