import json
from pathlib import Path

from sentarc_coding_agent.core.event_recorder import attach_event_recorder, create_event_log_path


class _DummyAgent:
    def __init__(self):
        self._listeners = []

    def subscribe(self, fn):
        self._listeners.append(fn)

        def _unsubscribe():
            self._listeners.remove(fn)

        return _unsubscribe

    def emit(self, event):
        for fn in list(self._listeners):
            fn(event)


def test_create_event_log_path():
    base = "/tmp/arc-test-events"
    path = create_event_log_path(base)
    assert path.startswith(base)
    assert path.endswith(".jsonl")
    assert "/events/" in path


def test_attach_event_recorder_writes_metadata_and_event(tmp_path):
    agent = _DummyAgent()
    out = Path(tmp_path) / "events" / "run.jsonl"

    attach_event_recorder(
        agent,
        str(out),
        metadata={"model": "gpt-4o", "provider": "openai"},
    )

    class Event:
        type = "turn_start"

        def __init__(self):
            self.value = 42

    agent.emit(Event())

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    m = json.loads(lines[0])
    e = json.loads(lines[1])

    assert m["recordType"] == "metadata"
    assert m["metadata"]["model"] == "gpt-4o"

    assert e["recordType"] == "event"
    assert e["eventType"] == "turn_start"
    assert e["event"]["value"] == 42
