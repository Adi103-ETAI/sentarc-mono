"""
Key constants and matching utilities for arc TUI components.

These constants correspond to the raw byte sequences produced by most
terminals.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Control characters
# ---------------------------------------------------------------------------
CTRL_A = "\x01"
CTRL_B = "\x02"
CTRL_C = "\x03"  # Interrupt / SIGINT
CTRL_D = "\x04"  # EOF
CTRL_E = "\x05"
CTRL_F = "\x06"
CTRL_K = "\x0b"
CTRL_L = "\x0c"  # Clear screen
CTRL_N = "\x0e"
CTRL_P = "\x10"
CTRL_U = "\x15"
CTRL_W = "\x17"
CTRL_Z = "\x1a"  # Suspend / SIGTSTP

# ---------------------------------------------------------------------------
# Common keys
# ---------------------------------------------------------------------------
ENTER = "\r"
NEWLINE = "\n"
ESC = "\x1b"
BACKSPACE = "\x7f"
TAB = "\t"

# ---------------------------------------------------------------------------
# Arrow keys (VT100 / xterm sequences)
# ---------------------------------------------------------------------------
UP = "\x1b[A"
DOWN = "\x1b[B"
RIGHT = "\x1b[C"
LEFT = "\x1b[D"

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
HOME = "\x1b[H"
END = "\x1b[F"
PAGE_UP = "\x1b[5~"
PAGE_DOWN = "\x1b[6~"
DELETE = "\x1b[3~"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def matches_key(data: str, key: str) -> bool:
    """
    Return True when *data* exactly matches *key*.

    Example::

        if matches_key(raw_input, CTRL_C):
            raise KeyboardInterrupt
    """
    return data == key
