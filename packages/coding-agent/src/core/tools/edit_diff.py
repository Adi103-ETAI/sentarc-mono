"""Edit-diff tool — applies unified diff patches to files."""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class StripBomResult:
    bom: str
    text: str


def strip_bom(text: str) -> StripBomResult:
    """Strip UTF-8 BOM from text."""
    if text.startswith("\ufeff"):
        return StripBomResult(bom="\ufeff", text=text[1:])
    return StripBomResult(bom="", text=text)


def detect_line_ending(text: str) -> str:
    """Detect dominant line ending: 'crlf', 'cr', or 'lf'."""
    crlf = text.count("\r\n")
    cr = text.count("\r") - crlf
    lf = text.count("\n") - crlf
    if crlf >= lf and crlf >= cr:
        return "crlf"
    if cr > lf:
        return "cr"
    return "lf"


def normalize_to_lf(text: str) -> str:
    """Normalize line endings to LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: str) -> str:
    """Restore original line endings."""
    if ending == "crlf":
        return text.replace("\n", "\r\n")
    elif ending == "cr":
        return text.replace("\n", "\r")
    return text


def normalize_for_fuzzy_match(text: str) -> str:
    """
    Normalize text for fuzzy matching:
    - Collapse runs of spaces/tabs to single space per line
    - Trim trailing whitespace per line
    """
    lines = text.split("\n")
    normalized = []
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).rstrip()
        normalized.append(line)
    return "\n".join(normalized)


@dataclass
class FuzzyMatchResult:
    found: bool
    index: int
    match_length: int
    content_for_replacement: str


def fuzzy_find_text(content: str, old_text: str) -> FuzzyMatchResult:
    """
    Find old_text in content. Tries exact match first, then fuzzy (whitespace-normalized).
    Returns FuzzyMatchResult with found=True if found.
    """
    # Exact match
    idx = content.find(old_text)
    if idx != -1:
        return FuzzyMatchResult(
            found=True,
            index=idx,
            match_length=len(old_text),
            content_for_replacement=content,
        )

    # Fuzzy match: normalize whitespace in both content and old_text
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old = normalize_for_fuzzy_match(old_text)
    idx = fuzzy_content.find(fuzzy_old)
    if idx != -1:
        return FuzzyMatchResult(
            found=True,
            index=idx,
            match_length=len(fuzzy_old),
            content_for_replacement=fuzzy_content,
        )

    return FuzzyMatchResult(found=False, index=-1, match_length=0, content_for_replacement=content)


@dataclass
class DiffResult:
    diff: str
    first_changed_line: Optional[int]


def generate_diff_string(old_content: str, new_content: str) -> DiffResult:
    """Generate a unified diff string between old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff_lines = list(difflib.unified_diff(old_lines, new_lines, lineterm=""))
    diff_str = "\n".join(diff_lines)

    # Find first changed line number in new file
    first_changed_line: Optional[int] = None
    for line in diff_lines:
        if line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            if m:
                first_changed_line = int(m.group(1))
                break

    return DiffResult(diff=diff_str, first_changed_line=first_changed_line)
