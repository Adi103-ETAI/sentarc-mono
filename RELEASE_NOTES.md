# Release Notes

## Version

Version: v0.1.4
Date: 2026-03-26
Branch basis: shadow

## Summary

This release focuses on runtime stability, provider compatibility hardening, safer shell-tool controls, improved observability, and better configuration ergonomics in the coding agent.

## Highlights

### 1. Provider robustness improvements (sentarc-ai)

- Hardened provider registration and alias resolution paths.
- Improved OpenAI and Google provider compatibility behavior.
- Added regression tests for critical provider and utility paths.

### 2. Coding-agent safety and observability (sentarc-coding-agent)

- Added optional JSONL event recorder for agent lifecycle events.
- Added bash security profile options:
  - standard
  - read-only
- Added support for custom bash block patterns.
- Added CLI flags:
  - --bash-security-profile
  - --event-log
  - --event-log-path

### 3. Layered settings behavior

- Settings now load from:
  - global file: ~/.arc/agent/settings.json
  - project file: .arc/settings.json
- Project settings override global settings field-by-field.

### 4. Compaction and session compatibility fixes

- Improved compaction call contract compatibility.
- Improved extraction behavior in assistant-content handling paths.

### 5. Documentation and test coverage upgrades

- README updates for security profile, event logging, and settings override behavior.
- Added focused tests for:
  - provider aliasing and validation
  - OpenAI provider behavior
  - CLI arg parsing additions
  - bash security modes
  - event recorder output
  - settings merge precedence

## Package-level Changelogs

- [packages/ai/CHANGELOG.md](packages/ai/CHANGELOG.md)
- [packages/agent/CHANGELOG.md](packages/agent/CHANGELOG.md)
- [packages/coding-agent/CHANGELOG.md](packages/coding-agent/CHANGELOG.md)
- [packages/tui/CHANGELOG.md](packages/tui/CHANGELOG.md)

## Upgrade Notes

- No breaking API changes intended in this release.
- If you rely on shell tooling behavior, verify the configured bash security profile in your environment.
- If you enable event logs, monitor file growth and rotate logs as needed.
- Project-level .arc/settings.json now overrides global settings where keys overlap.

## Validation Snapshot

- Branch comparison source: main..shadow
- Included commits:
  - c4d16b0
  - 238015e

## Known Follow-up Upgrades (Recommended Next)

1. Agent runtime hardening
- Propagate abort signals through all tool execution paths and add explicit cancellation tests.
- Add stronger concurrency guards around shared agent state in high-contention paths.

2. Tool safety and boundaries
- Tighten path-containment checks for file tools and add explicit traversal regression tests.
- Expand bash policy documentation with allowed/blocked command examples.

3. Coverage depth
- Increase tests for session manager migration/branching edge cases.
- Add integration tests for multi-provider capability parity and fallback behavior.

4. Release operations
- Add automated release validation gates (version sync checks + smoke install tests).

## Contributor-facing Note

This file reflects the published GitHub release notes for v0.1.4.
