"""Context compaction — summarises old messages to reduce token usage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sentarc_coding_agent.core.compaction.utils import (
    SUMMARIZATION_SYSTEM_PROMPT,
    calculate_context_tokens,
    compute_file_lists,
    create_file_ops,
    estimate_context_tokens,
    estimate_tokens,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)
from sentarc_coding_agent.core.messages import (
    convert_to_llm,
    create_branch_summary_message,
    create_compaction_summary_message,
    create_custom_message,
)

SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Progress
### Done
[List completed tasks with key decisions and outcomes]

### In Progress
[List tasks currently being worked on with current state]

## Key Files
[List important files that were read or modified, with brief descriptions]

## Context
[Important context, constraints, and technical decisions that the next LLM needs to know]

## Next Steps
[Clear action items for continuing the work]"""

UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE the Key Files section with any new files
- Keep the same format structure"""


@dataclass
class CompactionSettings:
    enabled: bool = True
    reserve_tokens: int = 16384
    keep_recent_tokens: int = 20000


DEFAULT_COMPACTION_SETTINGS = CompactionSettings()


@dataclass
class CompactionResult:
    summary: str
    first_kept_entry_id: str
    tokens_before: int
    details: Optional[Dict[str, Any]] = None


def get_last_assistant_usage(entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the last non-aborted assistant message usage."""
    from sentarc_coding_agent.core.compaction.utils import get_assistant_usage
    for entry in reversed(entries):
        if entry.get("type") == "message":
            msg = entry.get("message", {})
            usage = get_assistant_usage(msg)
            if usage:
                return usage
    return None


def _get_message_from_entry(entry: Dict[str, Any]) -> Optional[Any]:
    """Extract AgentMessage from a session entry."""
    etype = entry.get("type")
    if etype == "message":
        return entry.get("message")
    if etype == "custom_message":
        return create_custom_message(
            entry.get("customType", ""),
            entry.get("content", ""),
            entry.get("display", False),
            entry.get("details"),
            entry.get("timestamp", ""),
        )
    if etype == "branch_summary":
        return create_branch_summary_message(
            entry.get("summary", ""),
            entry.get("fromId", ""),
            entry.get("timestamp", ""),
        )
    if etype == "compaction":
        return create_compaction_summary_message(
            entry.get("summary", ""),
            entry.get("tokensBefore", 0),
            entry.get("timestamp", ""),
        )
    return None


def _extract_file_operations(
    messages: List[Any],
    entries: List[Dict[str, Any]],
    prev_compaction_index: int,
) -> Dict[str, Any]:
    """Extract file operations from messages and previous compaction entries."""
    file_ops = create_file_ops()

    if prev_compaction_index >= 0:
        prev = entries[prev_compaction_index]
        if not prev.get("fromHook") and prev.get("details"):
            details = prev["details"]
            if isinstance(details, dict):
                for f in details.get("readFiles", []):
                    file_ops["read"].add(f)
                for f in details.get("modifiedFiles", []):
                    file_ops["edited"].add(f)

    for msg in messages:
        extract_file_ops_from_message(msg, file_ops)

    return file_ops


def find_compact_point(
    entries: List[Dict[str, Any]],
    keep_recent_tokens: int,
) -> Optional[str]:
    """Find the entry ID that should be the first kept entry after compaction."""
    messages_with_entries = []
    for entry in entries:
        msg = _get_message_from_entry(entry)
        if msg:
            messages_with_entries.append((entry, msg))

    if not messages_with_entries:
        return None

    llm_messages = convert_to_llm([m for _, m in messages_with_entries])

    # Work backwards to find cut point
    accumulated_tokens = 0
    for i in range(len(llm_messages) - 1, -1, -1):
        msg_tokens = estimate_tokens(llm_messages[i])
        accumulated_tokens += msg_tokens
        if accumulated_tokens >= keep_recent_tokens:
            # Find the entry at this index
            if i < len(messages_with_entries):
                return messages_with_entries[i][0].get("id")

    # Keep all if under budget
    if messages_with_entries:
        return messages_with_entries[0][0].get("id")
    return None


async def compact_messages(
    entries: List[Dict[str, Any]],
    model: Any,
    settings: Optional[CompactionSettings] = None,
    custom_instructions: Optional[str] = None,
    previous_summary: Optional[str] = None,
) -> Optional[CompactionResult]:
    """
    Compact session entries into a summary.
    Returns CompactionResult or None if no compaction needed.
    """
    if settings is None:
        settings = DEFAULT_COMPACTION_SETTINGS

    if not settings.enabled:
        return None

    session_entries = [e for e in entries if e.get("type") != "session"]

    # Get messages for context
    messages_with_entries = []
    for entry in session_entries:
        msg = _get_message_from_entry(entry)
        if msg:
            messages_with_entries.append((entry, msg))

    if not messages_with_entries:
        return None

    llm_messages = convert_to_llm([m for _, m in messages_with_entries])
    context_estimate = estimate_context_tokens(llm_messages)
    tokens_before = context_estimate["tokens"]

    # Find previous compaction index
    prev_compaction_index = -1
    for i, entry in enumerate(session_entries):
        if entry.get("type") == "compaction" and not entry.get("fromHook"):
            prev_compaction_index = i

    all_messages = [m for _, m in messages_with_entries]
    file_ops = _extract_file_operations(all_messages, session_entries, prev_compaction_index)
    file_lists = compute_file_lists(file_ops)
    file_ops_text = format_file_operations(file_lists["readFiles"], file_lists["modifiedFiles"])

    # Find compaction point
    first_kept_id = find_compact_point(session_entries, settings.keep_recent_tokens)
    if not first_kept_id:
        return None

    # Get messages to summarize (before first_kept_id)
    to_summarize: List[Any] = []
    found = False
    for entry, msg in messages_with_entries:
        if entry.get("id") == first_kept_id:
            found = True
            break
        to_summarize.append(msg)

    if not to_summarize:
        return None

    # Generate summary via LLM
    conversation_text = serialize_conversation(convert_to_llm(to_summarize))

    prompt_suffix = UPDATE_SUMMARIZATION_PROMPT if previous_summary else SUMMARIZATION_PROMPT
    if custom_instructions:
        prompt_suffix = f"{prompt_suffix}\n\nAdditional focus: {custom_instructions}"

    if previous_summary:
        user_content = (
            f"{conversation_text}\n\n"
            f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
            f"{prompt_suffix}"
        )
    else:
        user_content = f"{conversation_text}\n\n{file_ops_text}\n\n{prompt_suffix}"

    try:
        from sentarc_ai.stream import complete_simple
        from sentarc_ai.types import Context, Message, Role

        llm_context = Context(
            messages=[Message(role=Role.USER, content=user_content)],
            system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        )

        result = await complete_simple(
            model=model,
            context=llm_context,
        )

        text_parts: List[str] = []
        for block in getattr(result, "content", []):
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", ""))

        summary = "\n".join(part for part in text_parts if part).strip()
        if not summary:
            summary = str(result)
        summary += file_ops_text
    except Exception as e:
        # Fallback: use a simple text summary
        summary = f"[Compaction failed: {e}]\n\nConversation had {len(to_summarize)} messages."

    return CompactionResult(
        summary=summary,
        first_kept_entry_id=first_kept_id,
        tokens_before=tokens_before,
        details={
            "readFiles": file_lists["readFiles"],
            "modifiedFiles": file_lists["modifiedFiles"],
        },
    )
