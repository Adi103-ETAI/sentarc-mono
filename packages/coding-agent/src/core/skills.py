"""Skills loader — reads skill definitions from the config directory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sentarc_coding_agent.config import CONFIG_DIR_NAME, get_agent_dir

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
IGNORE_FILE_NAMES = (".gitignore", ".ignore", ".fdignore")


@dataclass
class Skill:
    name: str
    description: str
    file_path: str
    base_dir: str
    source: str
    disable_model_invocation: bool


@dataclass
class ResourceDiagnostic:
    type: str  # "warning" | "collision"
    message: str
    path: str
    collision: Optional[Dict[str, Any]] = None


def _escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _validate_name(name: str, parent_dir_name: str) -> List[str]:
    errors = []
    if name != parent_dir_name:
        errors.append(f'name "{name}" does not match parent directory "{parent_dir_name}"')
    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"name exceeds {MAX_NAME_LENGTH} characters ({len(name)})")
    import re
    if not re.match(r'^[a-z0-9-]+$', name):
        errors.append("name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)")
    if name.startswith("-") or name.endswith("-"):
        errors.append("name must not start or end with a hyphen")
    if "--" in name:
        errors.append("name must not contain consecutive hyphens")
    return errors


def _validate_description(description: Optional[str]) -> List[str]:
    errors = []
    if not description or not description.strip():
        errors.append("description is required")
    elif len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(f"description exceeds {MAX_DESCRIPTION_LENGTH} characters ({len(description)})")
    return errors


def _should_ignore(path: str, root: str) -> bool:
    """Simple gitignore-style check using pathspec if available."""
    try:
        import pathspec
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        # Check root gitignore
        gi_path = os.path.join(root, ".gitignore")
        if os.path.isfile(gi_path):
            with open(gi_path, "r", encoding="utf-8", errors="replace") as f:
                spec = pathspec.PathSpec.from_lines("gitwildmatch", f.readlines())
            if spec.match_file(rel):
                return True
    except Exception:
        pass
    return False


def load_skill_from_file(
    file_path: str, source: str
) -> Tuple[Optional[Skill], List[ResourceDiagnostic]]:
    from sentarc_coding_agent.utils.frontmatter import parse_frontmatter
    diagnostics: List[ResourceDiagnostic] = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_content = f.read()
        fm = parse_frontmatter(raw_content)
        skill_dir = str(Path(file_path).parent)
        parent_dir_name = Path(skill_dir).name

        desc_errors = _validate_description(fm.get("description"))
        for err in desc_errors:
            diagnostics.append(ResourceDiagnostic(type="warning", message=err, path=file_path))

        name = fm.get("name") or parent_dir_name

        name_errors = _validate_name(name, parent_dir_name)
        for err in name_errors:
            diagnostics.append(ResourceDiagnostic(type="warning", message=err, path=file_path))

        if not fm.get("description") or not fm["description"].strip():
            return None, diagnostics

        return Skill(
            name=name,
            description=fm["description"],
            file_path=file_path,
            base_dir=skill_dir,
            source=source,
            disable_model_invocation=fm.get("disable-model-invocation", False) is True,
        ), diagnostics
    except Exception as e:
        diagnostics.append(ResourceDiagnostic(
            type="warning",
            message=str(e) or "failed to parse skill file",
            path=file_path,
        ))
        return None, diagnostics


def load_skills_from_dir(
    dir_path: str,
    source: str,
    include_root_files: bool = True,
    root: Optional[str] = None,
) -> Tuple[List[Skill], List[ResourceDiagnostic]]:
    skills: List[Skill] = []
    diagnostics: List[ResourceDiagnostic] = []

    if not os.path.exists(dir_path):
        return skills, diagnostics

    root_dir = root or dir_path

    try:
        entries = sorted(os.listdir(dir_path))
    except Exception:
        return skills, diagnostics

    # Check for SKILL.md in this directory first
    if "SKILL.md" in entries:
        full_path = os.path.join(dir_path, "SKILL.md")
        if os.path.isfile(full_path):
            skill, diags = load_skill_from_file(full_path, source)
            diagnostics.extend(diags)
            if skill:
                skills.append(skill)
            return skills, diagnostics

    for entry in entries:
        if entry.startswith(".") or entry == "node_modules":
            continue

        full_path = os.path.join(dir_path, entry)

        if os.path.isdir(full_path):
            sub_skills, sub_diags = load_skills_from_dir(full_path, source, False, root_dir)
            skills.extend(sub_skills)
            diagnostics.extend(sub_diags)
        elif include_root_files and os.path.isfile(full_path) and entry.endswith(".md"):
            skill, diags = load_skill_from_file(full_path, source)
            diagnostics.extend(diags)
            if skill:
                skills.append(skill)

    return skills, diagnostics


def load_skills(
    cwd: Optional[str] = None,
    agent_dir: Optional[str] = None,
    skill_paths: Optional[List[str]] = None,
    include_defaults: bool = True,
) -> Tuple[List[Skill], List[ResourceDiagnostic]]:
    """Load skills from all configured locations."""
    import os
    resolved_cwd = cwd or os.getcwd()
    resolved_agent_dir = agent_dir or get_agent_dir()

    skill_map: Dict[str, Skill] = {}
    real_path_set: set = set()
    all_diagnostics: List[ResourceDiagnostic] = []
    collision_diagnostics: List[ResourceDiagnostic] = []

    def add_skills(result: Tuple[List[Skill], List[ResourceDiagnostic]]) -> None:
        new_skills, diags = result
        all_diagnostics.extend(diags)
        for skill in new_skills:
            try:
                real_path = os.path.realpath(skill.file_path)
            except Exception:
                real_path = skill.file_path

            if real_path in real_path_set:
                continue

            existing = skill_map.get(skill.name)
            if existing:
                collision_diagnostics.append(ResourceDiagnostic(
                    type="collision",
                    message=f'name "{skill.name}" collision',
                    path=skill.file_path,
                    collision={
                        "resourceType": "skill",
                        "name": skill.name,
                        "winnerPath": existing.file_path,
                        "loserPath": skill.file_path,
                    },
                ))
            else:
                skill_map[skill.name] = skill
                real_path_set.add(real_path)

    if include_defaults:
        add_skills(load_skills_from_dir(
            os.path.join(resolved_agent_dir, "skills"), "user"
        ))
        add_skills(load_skills_from_dir(
            os.path.join(resolved_cwd, CONFIG_DIR_NAME, "skills"), "project"
        ))

    for raw_path in (skill_paths or []):
        p = raw_path.strip()
        if p.startswith("~/"):
            p = str(Path.home() / p[2:])
        elif p == "~":
            p = str(Path.home())

        if not os.path.isabs(p):
            p = os.path.join(resolved_cwd, p)

        if not os.path.exists(p):
            all_diagnostics.append(ResourceDiagnostic(
                type="warning", message="skill path does not exist", path=p
            ))
            continue

        if os.path.isdir(p):
            add_skills(load_skills_from_dir(p, "path"))
        elif os.path.isfile(p) and p.endswith(".md"):
            skill, diags = load_skill_from_file(p, "path")
            if skill:
                add_skills(([skill], diags))
            else:
                all_diagnostics.extend(diags)
        else:
            all_diagnostics.append(ResourceDiagnostic(
                type="warning", message="skill path is not a markdown file", path=p
            ))

    return list(skill_map.values()), [*all_diagnostics, *collision_diagnostics]


def format_skills_for_prompt(skills: List[Skill]) -> str:
    """Format skills for inclusion in a system prompt (XML format)."""
    visible = [s for s in skills if not s.disable_model_invocation]
    if not visible:
        return ""

    lines = [
        "\n\nThe following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
        "",
        "<available_skills>",
    ]

    for skill in visible:
        lines.append("  <skill>")
        lines.append(f"    <name>{_escape_xml(skill.name)}</name>")
        lines.append(f"    <description>{_escape_xml(skill.description)}</description>")
        lines.append(f"    <location>{_escape_xml(skill.file_path)}</location>")
        lines.append("  </skill>")

    lines.append("</available_skills>")
    return "\n".join(lines)
