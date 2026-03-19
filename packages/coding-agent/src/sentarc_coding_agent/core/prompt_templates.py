"""Load prompt templates from ~/.arc/agent/prompts/."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sentarc_coding_agent.config import get_agent_dir, CONFIG_DIR_NAME


@dataclass
class PromptTemplate:
    name: str
    content: str
    file_path: str
    description: Optional[str] = None


def load_prompt_template_from_file(
    file_path: str,
) -> Tuple[Optional[PromptTemplate], List[str]]:
    """Load a prompt template from a markdown/text file with optional frontmatter."""
    from sentarc_coding_agent.utils.frontmatter import parse_frontmatter
    warnings: List[str] = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        fm = parse_frontmatter(raw)
        name = fm.get("name") or Path(file_path).stem
        description = fm.get("description")
        # Remove frontmatter from content
        body = raw
        if raw.startswith("---"):
            end = raw.find("---", 3)
            if end != -1:
                body = raw[end + 3:].lstrip("\n")
        return PromptTemplate(
            name=name,
            content=body,
            file_path=file_path,
            description=description,
        ), warnings
    except Exception as e:
        warnings.append(f"Failed to load prompt template {file_path}: {e}")
        return None, warnings


def load_prompt_templates(
    cwd: Optional[str] = None,
    agent_dir: Optional[str] = None,
    template_paths: Optional[List[str]] = None,
    include_defaults: bool = True,
) -> Tuple[List[PromptTemplate], List[str]]:
    """Load prompt templates from all configured locations."""
    resolved_cwd = cwd or os.getcwd()
    resolved_agent_dir = agent_dir or get_agent_dir()

    templates: List[PromptTemplate] = []
    all_warnings: List[str] = []

    def load_from_dir(dir_path: str) -> None:
        if not os.path.isdir(dir_path):
            return
        for fname in sorted(os.listdir(dir_path)):
            if fname.endswith(".md") or fname.endswith(".txt"):
                t, warnings = load_prompt_template_from_file(os.path.join(dir_path, fname))
                all_warnings.extend(warnings)
                if t:
                    templates.append(t)

    if include_defaults:
        load_from_dir(os.path.join(resolved_agent_dir, "prompts"))
        load_from_dir(os.path.join(resolved_cwd, CONFIG_DIR_NAME, "prompts"))

    for raw_path in (template_paths or []):
        p = os.path.expanduser(raw_path)
        if not os.path.isabs(p):
            p = os.path.join(resolved_cwd, p)
        if os.path.isdir(p):
            load_from_dir(p)
        elif os.path.isfile(p):
            t, warnings = load_prompt_template_from_file(p)
            all_warnings.extend(warnings)
            if t:
                templates.append(t)

    return templates, all_warnings
