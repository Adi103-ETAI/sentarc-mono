# sentarc-coding-agent Changelog

All notable changes to sentarc-coding-agent are documented in this file.

## 0.1.4 (planned)
Date: 2026-03-26

### Added

- Optional event observability recorder:
  - src/sentarc_coding_agent/core/event_recorder.py
- New tests for CLI flags, bash policy, compaction, event logging, and settings merge:
  - tests/test_args.py
  - tests/test_bash_security.py
  - tests/test_compaction.py
  - tests/test_event_recorder.py
  - tests/test_settings_manager.py

### Changed

- CLI/runtime wiring for optional event logging:
  - src/sentarc_coding_agent/cli/__init__.py
  - src/sentarc_coding_agent/core/agent_session.py
- CLI argument parser now supports:
  - --bash-security-profile
  - --event-log
  - --event-log-path
- Settings loading supports layered configuration:
  - global: ~/.arc/agent/settings.json
  - project: .arc/settings.json
  - project values override global values field-by-field.
- Bash tool now supports policy modes and custom blocked patterns.
- Bash abort behavior tightened:
  - Lower abort latency via event-driven wait.
  - TERM to KILL escalation for stubborn process trees.
- Tool factory plumbing now forwards bash policy options consistently.
- Compaction contract and extraction behavior improved for compatibility.

### Fixed

- Prompt/session contract mismatches in fallback session flow.
- Documentation/runtime mismatches in README and CLI help text.
- Path traversal hardening in path resolution:
  - Absolute and relative paths are now containment-checked against cwd.
  - Read-path variant candidates are re-validated before use.

### Added

- Additional traversal regression tests in:
  - tests/test_path_utils.py
- Additional bash abort regression test in:
  - tests/test_bash_security.py

### Documentation

- README updates for bash security profile, event logs, and project-level settings overrides.

### Notes

- Version header is marked planned until package version is bumped and published.
