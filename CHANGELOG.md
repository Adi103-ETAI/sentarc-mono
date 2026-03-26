# Changelog

All notable changes in this repository are documented in this file.

Per-package changelogs:

- [packages/ai/CHANGELOG.md](packages/ai/CHANGELOG.md)
- [packages/agent/CHANGELOG.md](packages/agent/CHANGELOG.md)
- [packages/coding-agent/CHANGELOG.md](packages/coding-agent/CHANGELOG.md)
- [packages/tui/CHANGELOG.md](packages/tui/CHANGELOG.md)

## 2026-03-26 (shadow vs main)

Source: commits `c4d16b0` and `238015e` on branch `shadow`.

### Added

- Root-level provider alias and registration validation tests:
  - `packages/ai/tests/test_registry_alias.py`
- OpenAI provider regression tests:
  - `packages/ai/tests/test_openai_completions_provider.py`
  - `packages/ai/tests/test_openai_responses_provider.py`
  - `packages/ai/tests/test_openai_utils.py`
- Coding-agent observability recorder:
  - `packages/coding-agent/src/sentarc_coding_agent/core/event_recorder.py`
- Coding-agent test coverage for newly introduced behavior:
  - `packages/coding-agent/tests/test_args.py`
  - `packages/coding-agent/tests/test_bash_security.py`
  - `packages/coding-agent/tests/test_compaction.py`
  - `packages/coding-agent/tests/test_event_recorder.py`
  - `packages/coding-agent/tests/test_settings_manager.py`
- Additional safety regression tests:
  - `packages/coding-agent/tests/test_path_utils.py`
  - `packages/agent/tests/test_agent.py` (abort during slow stream)
  - `packages/coding-agent/tests/test_bash_security.py` (fast abort regression)

### Changed

- Provider registry hardening in `packages/ai/src/sentarc_ai/registry.py`:
  - Added provider API alias resolution support.
  - Added fail-fast validation for provider registration (api name and callable handlers).
- OpenAI/Google provider compatibility and safety updates:
  - `packages/ai/src/sentarc_ai/providers/openai.py`
  - `packages/ai/src/sentarc_ai/providers/openai_codex.py`
  - `packages/ai/src/sentarc_ai/providers/openai_completions.py`
  - `packages/ai/src/sentarc_ai/providers/openai_responses.py`
  - `packages/ai/src/sentarc_ai/providers/openai_utils.py`
  - `packages/ai/src/sentarc_ai/providers/google.py`
- Coding-agent CLI and runtime wiring:
  - `packages/coding-agent/src/sentarc_coding_agent/cli/__init__.py`
  - `packages/coding-agent/src/sentarc_coding_agent/cli/args.py`
  - `packages/coding-agent/src/sentarc_coding_agent/core/agent_session.py`
  - Added bash security profile options and event-log options in CLI/runtime.
- Settings behavior in `packages/coding-agent/src/sentarc_coding_agent/core/settings_manager.py`:
  - Added merged settings loading from global and project scope.
  - Project file `.arc/settings.json` now overrides global settings field-by-field.
- Tool construction updates in `packages/coding-agent/src/sentarc_coding_agent/core/tools/__init__.py`:
  - Centralized bash tool option plumbing for security profile and block patterns.
- Bash tool policy extension in `packages/coding-agent/src/sentarc_coding_agent/core/tools/bash.py`:
  - Added security profile modes (`standard`, `read-only`).
  - Added optional custom regex block patterns.
- Compaction flow update in `packages/coding-agent/src/sentarc_coding_agent/core/compaction/compaction.py`.
- Path containment hardening in `packages/coding-agent/src/sentarc_coding_agent/core/tools/path_utils.py`.
- Abort handling improvements in `packages/agent/src/sentarc_agent/agent_loop.py`.
- Bash kill escalation and abort-latency tightening in `packages/coding-agent/src/sentarc_coding_agent/core/tools/bash.py`.

### Fixed

- Prompt/session contract mismatches in coding-agent runtime flow.
- Provider option parsing issues and API key precedence behavior in OpenAI provider paths.
- Compaction invocation contract and assistant text extraction reliability.
- Documentation/runtime mismatches in settings and CLI help text.

### Documentation

- Updated `packages/coding-agent/README.md` with:
  - bash security profile usage,
  - event logging usage,
  - project-level settings override behavior,
  - settings schema examples and guidance.

### Notes

- This changelog entry is based on `git diff --name-status main..shadow` at the time of writing.
- Recommended next release notes split: `Phase 1 stability`, `Phase 2 safety/observability`, `Docs + tests`.
