"""Path utilities — resolves and normalises file paths for tool operations."""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

_UNICODE_SPACES = "\u00A0\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200A\u202F\u205F\u3000"
_NARROW_NO_BREAK_SPACE = "\u202F"


def _normalize_unicode_spaces(s: str) -> str:
    for ch in _UNICODE_SPACES:
        s = s.replace(ch, " ")
    return s


def _normalize_at_prefix(s: str) -> str:
    return s[1:] if s.startswith("@") else s


def _try_macos_screenshot_path(file_path: str) -> str:
    import re
    return re.sub(r" (AM|PM)\.", f"{_NARROW_NO_BREAK_SPACE}\\1.", file_path)


def _try_nfd_variant(file_path: str) -> str:
    return unicodedata.normalize("NFD", file_path)


def _try_curly_quote_variant(file_path: str) -> str:
    return file_path.replace("'", "\u2019")


def _file_exists(file_path: str) -> bool:
    return os.path.exists(file_path)


def expand_path(file_path: str) -> str:
    """Expand ~ and normalize unicode spaces in a path."""
    normalized = _normalize_unicode_spaces(_normalize_at_prefix(file_path))
    if normalized == "~":
        return str(Path.home())
    if normalized.startswith("~/"):
        return str(Path.home() / normalized[2:])
    return normalized


def resolve_to_cwd(file_path: str, cwd: str) -> str:
    """Resolve a path relative to the given cwd and enforce containment."""
    expanded = expand_path(file_path)
    cwd_resolved = Path(cwd).resolve()

    if os.path.isabs(expanded):
        resolved = Path(expanded).resolve()
    else:
        resolved = (cwd_resolved / expanded).resolve()

    try:
        resolved.relative_to(cwd_resolved)
    except ValueError:
        raise Exception(
            f"Path traversal detected: {file_path} resolves outside working directory.\n"
            f"Resolved: {resolved}\n"
            f"Working directory: {cwd_resolved}"
        )

    return str(resolved)


def resolve_read_path(file_path: str, cwd: str) -> str:
    """Resolve a read path, trying macOS-specific variants if needed."""
    resolved = resolve_to_cwd(file_path, cwd)

    if _file_exists(resolved):
        return resolved

    # Try macOS AM/PM variant
    am_pm_variant = _try_macos_screenshot_path(resolved)
    if am_pm_variant != resolved and _file_exists(am_pm_variant):
        resolve_to_cwd(am_pm_variant, cwd)
        return am_pm_variant

    # Try NFD variant
    nfd_variant = _try_nfd_variant(resolved)
    if nfd_variant != resolved and _file_exists(nfd_variant):
        resolve_to_cwd(nfd_variant, cwd)
        return nfd_variant

    # Try curly quote variant
    curly_variant = _try_curly_quote_variant(resolved)
    if curly_variant != resolved and _file_exists(curly_variant):
        resolve_to_cwd(curly_variant, cwd)
        return curly_variant

    # Try combined NFD + curly quote
    nfd_curly_variant = _try_curly_quote_variant(nfd_variant)
    if nfd_curly_variant != resolved and _file_exists(nfd_curly_variant):
        resolve_to_cwd(nfd_curly_variant, cwd)
        return nfd_curly_variant

    return resolved
