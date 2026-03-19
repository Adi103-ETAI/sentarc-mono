"""
Utility functions for terminal text manipulation.
"""
from __future__ import annotations

import re
import wcwidth as _wcwidth

# Regex that matches ANSI/VT100 escape sequences
_ANSI_RE = re.compile(
    r"""
    \x1b   # ESC
    (?:
        \[  # CSI sequences
        [0-9;?]*
        [A-Za-z]
    |
        \]  # OSC sequences (e.g., hyperlinks)
        [^\x07\x1b]*
        (?:\x07|\x1b\\)
    |
        [PX^_]  # DCS / SOS / PM / APC
        [^\x1b]*
        \x1b\\
    |
        .   # all other two-character escape sequences
    )
""",
    re.VERBOSE,
)


def strip_ansi(text: str) -> str:
    """Remove all ANSI escape sequences from *text*."""
    return _ANSI_RE.sub("", text)


def visible_width(text: str) -> int:
    """
    Return the *visible* terminal column width of *text*.

    Strips ANSI codes first, then sums per-character widths via wcwidth
    (handles CJK double-width characters correctly).
    """
    clean = strip_ansi(text)
    width = 0
    for ch in clean:
        w = _wcwidth.wcwidth(ch)
        if w > 0:
            width += w
    return width


def truncate_to_width(text: str, max_width: int) -> str:
    """
    Truncate *text* so that its visible terminal width does not exceed *max_width*.

    ANSI escape sequences are preserved and do not count toward the width.
    The function walks through the original string character-by-character,
    accumulating visible width, and stops as soon as the limit is reached.
    """
    if max_width <= 0:
        return ""

    # We need to walk both the raw string (to preserve ANSI codes) and track width.
    result: list[str] = []
    current_width = 0
    i = 0

    while i < len(text):
        # Check for ANSI escape sequence starting at position i
        m = _ANSI_RE.match(text, i)
        if m:
            result.append(m.group())
            i = m.end()
            continue

        ch = text[i]
        w = _wcwidth.wcwidth(ch)
        if w < 0:
            w = 0  # non-printable — include but don't count
        if current_width + w > max_width:
            break
        result.append(ch)
        current_width += w
        i += 1

    return "".join(result)
