# sentarc-ai Changelog

All notable changes to sentarc-ai are documented in this file.

## 0.1.4 (planned)
Date: 2026-03-26

### Added

- Tests for OpenAI provider compatibility and regression handling:
  - tests/test_openai_completions_provider.py
  - tests/test_openai_responses_provider.py
  - tests/test_openai_utils.py
  - tests/test_registry_alias.py

### Changed

- Hardened provider registry behavior:
  - API alias resolution coverage and behavior validation.
  - Fail-fast provider registration checks (non-empty api, callable stream handlers).
- Updated provider implementations for runtime/type compatibility:
  - src/sentarc_ai/providers/openai.py
  - src/sentarc_ai/providers/openai_codex.py
  - src/sentarc_ai/providers/openai_completions.py
  - src/sentarc_ai/providers/openai_responses.py
  - src/sentarc_ai/providers/openai_utils.py
  - src/sentarc_ai/providers/google.py

### Fixed

- API key precedence and options parsing paths in OpenAI provider flows.
- Token usage and event parsing mismatches that could cause runtime inconsistencies.

### Notes

- Version header is marked planned until package version is bumped and published.
