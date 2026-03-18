# Examples

Ready-made extensions and skills for sentarc-coding-agent. Run them with `arc -e <path>` and point skills at `examples/skills/` via `--skill examples/skills`.

## SDK
- `sdk/` — programmatic Python examples (Agent basics, tools, sessions)

## Extensions
- `extensions/hello.py` — minimal hook plus a greeting tool
- `extensions/commands.py` — registers slash commands
- `extensions/permission-gate.py` — blocks dangerous bash strings
- `extensions/protected-paths.py` — prevents writes to protected paths
- `extensions/git-checkpoint.py` — optional git stash checkpoints
- `extensions/auto-commit.py` — optional auto-commit on session end
- `extensions/custom-footer.py` — shows session stats in the interactive UI
- `extensions/notify.py` — desktop notifications for assistant replies

## Skills
- `skills/code-review/` — lightweight review checklist
- `skills/git-workflow/` — common git hygiene steps
