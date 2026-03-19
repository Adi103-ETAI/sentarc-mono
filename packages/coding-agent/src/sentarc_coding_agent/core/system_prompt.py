"""System prompt builder — constructs the agent's system prompt from tools, skills, and context."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from sentarc_coding_agent.config import get_readme_path, get_docs_path, get_examples_path

_TOOL_DESCRIPTIONS: Dict[str, str] = {
    "read": "Read file contents",
    "bash": "Execute bash commands (ls, grep, find, etc.)",
    "edit": "Make surgical edits to files (find exact text and replace)",
    "write": "Create or overwrite files",
    "grep": "Search file contents for patterns (respects .gitignore)",
    "find": "Find files by glob pattern (respects .gitignore)",
    "ls": "List directory contents",
}


def build_system_prompt(
    custom_prompt: Optional[str] = None,
    selected_tools: Optional[List[str]] = None,
    tool_snippets: Optional[Dict[str, str]] = None,
    prompt_guidelines: Optional[List[str]] = None,
    append_system_prompt: Optional[str] = None,
    cwd: Optional[str] = None,
    context_files: Optional[List[Dict[str, str]]] = None,
    skills: Optional[List[Any]] = None,
) -> str:
    """Build the system prompt with tools, guidelines, and context."""
    import os
    resolved_cwd = cwd or os.getcwd()
    prompt_cwd = resolved_cwd.replace("\\", "/")
    today = date.today().isoformat()
    append_section = f"\n\n{append_system_prompt}" if append_system_prompt else ""
    context_files = context_files or []
    skills = skills or []

    if custom_prompt:
        prompt = custom_prompt
        if append_section:
            prompt += append_section

        if context_files:
            prompt += "\n\n# Project Context\n\n"
            prompt += "Project-specific instructions and guidelines:\n\n"
            for cf in context_files:
                prompt += f"## {cf['path']}\n\n{cf['content']}\n\n"

        has_read = not selected_tools or "read" in selected_tools
        if has_read and skills:
            from sentarc_coding_agent.core.skills import format_skills_for_prompt
            prompt += format_skills_for_prompt(skills)

        prompt += f"\nCurrent date: {today}"
        prompt += f"\nCurrent working directory: {prompt_cwd}"
        return prompt

    readme_path = get_readme_path()
    docs_path = get_docs_path()
    examples_path = get_examples_path()

    tools = selected_tools or ["read", "bash", "edit", "write"]

    tools_list_lines = []
    for name in tools:
        snippet = (tool_snippets or {}).get(name) or _TOOL_DESCRIPTIONS.get(name) or name
        tools_list_lines.append(f"- {name}: {snippet}")
    tools_list = "\n".join(tools_list_lines) if tools_list_lines else "(none)"

    # Build guidelines
    guidelines_set: set = set()
    guidelines_list: List[str] = []

    def add_guideline(g: str) -> None:
        if g not in guidelines_set:
            guidelines_set.add(g)
            guidelines_list.append(g)

    has_bash = "bash" in tools
    has_edit = "edit" in tools
    has_write = "write" in tools
    has_grep = "grep" in tools
    has_find = "find" in tools
    has_ls = "ls" in tools
    has_read = "read" in tools

    if has_bash and not has_grep and not has_find and not has_ls:
        add_guideline("Use bash for file operations like ls, rg, find")
    elif has_bash and (has_grep or has_find or has_ls):
        add_guideline("Prefer grep/find/ls tools over bash for file exploration (faster, respects .gitignore)")

    if has_read and has_edit:
        add_guideline("Use read to examine files before editing. You must use this tool instead of cat or sed.")

    if has_edit:
        add_guideline("Use edit for precise changes (old text must match exactly)")

    if has_write:
        add_guideline("Use write only for new files or complete rewrites")

    if has_edit or has_write:
        add_guideline(
            "When summarizing your actions, output plain text directly - do NOT use cat or bash to display what you did"
        )

    for guideline in (prompt_guidelines or []):
        normalized = guideline.strip()
        if normalized:
            add_guideline(normalized)

    add_guideline("Be concise in your responses")
    add_guideline("Show file paths clearly when working with files")

    guidelines = "\n".join(f"- {g}" for g in guidelines_list)

    prompt = f"""You are an expert coding assistant operating inside arc, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
{tools_list}

In addition to the tools above, you may have access to other custom tools depending on the project.

Guidelines:
{guidelines}

Arc documentation (read only when the user asks about arc itself, its SDK, extensions, themes, skills, or TUI):
- Main documentation: {readme_path}
- Additional docs: {docs_path}
- Examples: {examples_path} (extensions, custom tools, SDK)
- When asked about: extensions (docs/extensions.md, examples/extensions/), themes (docs/themes.md), skills (docs/skills.md), prompt templates (docs/prompt-templates.md), TUI components (docs/tui.md), keybindings (docs/keybindings.md), SDK integrations (docs/sdk.md), custom providers (docs/custom-provider.md), adding models (docs/models.md), arc packages (docs/packages.md)
- When working on arc topics, read the docs and examples, and follow .md cross-references before implementing
- Always read arc .md files completely and follow links to related docs (e.g., tui.md for TUI API details)"""

    if append_section:
        prompt += append_section

    if context_files:
        prompt += "\n\n# Project Context\n\n"
        prompt += "Project-specific instructions and guidelines:\n\n"
        for cf in context_files:
            prompt += f"## {cf['path']}\n\n{cf['content']}\n\n"

    if has_read and skills:
        from sentarc_coding_agent.core.skills import format_skills_for_prompt
        prompt += format_skills_for_prompt(skills)

    prompt += f"\nCurrent date: {today}"
    prompt += f"\nCurrent working directory: {prompt_cwd}"

    return prompt
