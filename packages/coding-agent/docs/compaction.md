# Compaction

The compaction subsystem (`sentarc_coding_agent.core.compaction`) summarizes older conversation spans so new prompts stay within a model's context window without losing intent.

## Settings

`CompactionSettings` defines the available knobs:

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `True` | Disable to bypass compaction entirely. |
| `reserve_tokens` | `16384` | Reserved space for downstream consumers (currently informational in the Python build). |
| `keep_recent_tokens` | `20000` | Number of most recent tokens to keep verbatim before summarizing earlier history. |

Use `DEFAULT_COMPACTION_SETTINGS` as-is or clone it before overriding values per provider/model.

## How `compact_messages()` works

1. Filter out the `session` header and convert the remaining entries into Agent messages via `_get_message_from_entry()`.
2. Estimate token usage (`estimate_context_tokens`) and remember the pre-compaction count.
3. Look for the most recent `compaction` entry to reuse prior file-operation metadata when available.
4. Compute file I/O summaries by merging `_extract_file_operations()` with `compute_file_lists()`; this produces `readFiles` and `modifiedFiles` arrays.
5. Call `find_compact_point()` to choose the earliest entry that should remain uncompressed based on `keep_recent_tokens`.
6. Serialize everything before that entry (`serialize_conversation`), attach file operations, and build either `SUMMARIZATION_PROMPT` or `UPDATE_SUMMARIZATION_PROMPT` depending on whether `previous_summary` was provided.
7. Ask `sentarc_ai.stream.complete_simple()` (using `SUMMARIZATION_SYSTEM_PROMPT`) to generate the structured summary. On failure, fall back to an inline error note.

The function returns `CompactionResult(summary, first_kept_entry_id, tokens_before, details)` so callers can persist the summary and show metadata.

## Persisting results

Invoke `SessionManager.append_compaction(result.summary, result.first_kept_entry_id, result.tokens_before, result.details, from_hook=False)` to write the summary back to the JSONL file. Entries marked with `fromHook=True` let you differentiate automated compaction (extensions, schedulers) from user-triggered actions.

## Manual triggers today

- `/compact [notes]` currently replies with a placeholder string, so use extensions or RPC tooling to call `compact_messages()` directly when you detect context pressure.
- Extensions can monitor `session_manager.get_entries()`, run compaction asynchronously, then append summaries and branch markers via the standard APIs.

This design keeps the full history in the JSONL file for auditing while giving the model a concise `Goal / Progress / Key Files / Context / Next Steps` checkpoint to continue from.
