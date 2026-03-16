"""diff — /diff command shows git changes and opens selected file in editor."""

import subprocess
from typing import Any, List

EXTENSION_NAME = "diff"


async def _diff_handler(args: str, ctx: Any) -> str:
    """Show modified/deleted/new files from git status."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=ctx.cwd,
    )
    if result.returncode != 0:
        return f"git status failed: {result.stderr.strip()}"

    output = result.stdout.strip()
    if not output:
        return "No changes in working tree."

    lines: List[str] = []
    for line in output.splitlines():
        if len(line) < 4:
            continue
        status = line[:2].strip() or "~"
        filename = line[2:].strip()

        # Map status codes to labels
        if "M" in line[:2]:
            label = "M"
        elif "A" in line[:2]:
            label = "A"
        elif "D" in line[:2]:
            label = "D"
        elif "?" in line[:2]:
            label = "?"
        elif "R" in line[:2]:
            label = "R"
        elif "C" in line[:2]:
            label = "C"
        else:
            label = status

        lines.append(f"  {label} {filename}")

    header = f"Changed files ({len(lines)}):"
    return header + "\n" + "\n".join(lines)


COMMANDS = [
    {
        "name": "diff",
        "description": "Show git changes",
        "handler": _diff_handler,
    },
]
