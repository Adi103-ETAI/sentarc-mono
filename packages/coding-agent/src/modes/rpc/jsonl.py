"""Newline-delimited JSON I/O utilities."""
from __future__ import annotations

import json
import sys
from typing import Any


def write_jsonl(data: Any) -> None:
    """Write object as JSON line to stdout."""
    print(json.dumps(data, default=str), flush=True)


def read_jsonl_line(line: str) -> Any:
    """Parse a JSON line."""
    return json.loads(line.strip())


class JsonlReader:
    """Async reader for JSON lines from stdin."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdin

    async def __aiter__(self):
        import asyncio
        loop = asyncio.get_event_loop()
        while True:
            line = await loop.run_in_executor(None, self.stream.readline)
            if not line:
                break
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass
