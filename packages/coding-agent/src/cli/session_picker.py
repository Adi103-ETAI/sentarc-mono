"""Interactive session selection for --resume flag."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def select_session_interactive(sessions_dir: Optional[Path] = None) -> Optional[str]:
    """
    Show list of recent sessions and let user pick one.
    Returns session ID or None if cancelled.
    """
    from sentarc_coding_agent.core.session_manager import SessionManager
    import asyncio
    import os

    cwd = os.getcwd()
    # Use the default session dir for the cwd
    from sentarc_coding_agent.core.session_manager import get_default_session_dir, list_sessions_from_dir
    sdir = str(sessions_dir) if sessions_dir else get_default_session_dir(cwd)

    try:
        sessions = asyncio.run(list_sessions_from_dir(sdir))
    except Exception:
        sessions = []

    if not sessions:
        print("No sessions found.")
        return None

    print("\nAvailable sessions:")
    shown = sessions[:20]
    for i, s in enumerate(shown):
        ts = s.get("timestamp", "")[:19] if isinstance(s, dict) else ""
        sid = s.get("id", "")[:8] if isinstance(s, dict) else str(s)[:8]
        scwd = s.get("cwd", "") if isinstance(s, dict) else ""
        print(f"  {i + 1}. {ts}  {sid}  {scwd}")

    try:
        choice = input(f"\nSelect session (1-{len(shown)}) or Enter to cancel: ").strip()
        if not choice:
            return None
        idx = int(choice) - 1
        if 0 <= idx < len(shown):
            s = shown[idx]
            return s.get("id") if isinstance(s, dict) else None
    except (ValueError, KeyboardInterrupt, EOFError):
        pass
    return None
