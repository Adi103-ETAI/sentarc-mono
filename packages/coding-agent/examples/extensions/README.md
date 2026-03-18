# Extension examples

Drop these files into `~/.arc/agent/tools/` or load directly:

```bash
arc -e examples/extensions/hello.py
```

| File | Purpose |
| --- | --- |
| `hello.py` | Minimal extension with a greeting tool and `/hello` command |
| `commands.py` | Registers multiple slash commands |
| `permission-gate.py` | Blocks dangerous bash invocations |
| `protected-paths.py` | Refuses writes/edits in protected directories |
| `git-checkpoint.py` | Optional git stash after each run (opt-in env) |
| `auto-commit.py` | Optional auto-commit on session end (opt-in env) |
| `custom-footer.py` | Shows branch/message stats in the interactive footer |
| `notify.py` | Sends desktop notifications for assistant messages |
