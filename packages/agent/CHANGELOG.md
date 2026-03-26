# sentarc-agent Changelog

All notable changes to sentarc-agent are documented in this file.

## 0.1.4 (planned)
Date: 2026-03-26

### Changed

- Agent loop abort handling is now more responsive during streaming and tool execution.
- Main loop now exits earlier when abort is requested.

### Added

- Regression test for aborting a slow streaming prompt:
	- tests/test_agent.py

### Notes

- Keep this file as the source of truth for future agent-loop, event, and tool execution changes.
- Version header is marked planned until package version is bumped and published.
