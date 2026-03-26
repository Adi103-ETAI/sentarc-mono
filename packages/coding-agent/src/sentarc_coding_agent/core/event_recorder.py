"""Agent event recorder for optional JSONL observability logs."""

from __future__ import annotations

import json
import os
from dataclasses import is_dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if hasattr(value, "__dict__"):
        data = {
            k: v
            for k, v in vars(value).items()
            if not k.startswith("_") and not callable(v)
        }
        return _to_jsonable(data)
    return str(value)


def create_event_log_path(base_dir: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return str(Path(base_dir) / "events" / f"event-log-{ts}.jsonl")


def attach_event_recorder(
    agent: Any,
    file_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """Attach a listener that writes agent events as JSONL records."""

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if metadata:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "recordType": "metadata",
                        "metadata": _to_jsonable(metadata),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _on_event(event: Any) -> None:
        event_type = getattr(event, "type", event.__class__.__name__)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recordType": "event",
            "eventType": event_type,
            "event": _to_jsonable(event),
        }
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return agent.subscribe(_on_event)
