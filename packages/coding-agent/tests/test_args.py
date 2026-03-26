from sentarc_coding_agent.cli.args import parse_args


def test_parse_args_event_log_flags():
    args = parse_args(["--event-log", "--event-log-path", "/tmp/arc-events.jsonl", "hello"])
    assert args.get("event_log") is True
    assert args.get("event_log_path") == "/tmp/arc-events.jsonl"
    assert args.get("messages") == ["hello"]
