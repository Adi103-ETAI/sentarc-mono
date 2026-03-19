"""Bash tool — executes shell commands with streaming output and timeout support."""

from __future__ import annotations

import asyncio
import os
import re
import signal as _signal
import tempfile
from typing import Any, Callable, Dict, List, Optional

from sentarc_coding_agent.core.tools.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    format_size,
    truncate_tail,
)


# Dangerous command patterns that could cause system damage
DANGEROUS_PATTERNS = [
    (r'rm\s+-rf\s+/', "Recursive deletion of root directory"),
    (r'rm\s+-rf\s+\*', "Recursive deletion with wildcard"),
    (r':\(\)\{\s*:\|:&\s*\};:', "Fork bomb pattern"),
    (r'dd\s+if=.*\s+of=/dev/(sd|hd|nvme)', "Writing directly to disk device"),
    (r'mkfs\.', "Filesystem formatting command"),
    (r'>\s*/dev/(sd|hd|nvme)', "Redirecting output to disk device"),
    (r'chmod\s+-R\s+777\s+/', "Changing permissions on root directory"),
    (r'curl.*\|\s*(sh|bash|python)', "Piping download directly to interpreter"),
    (r'wget.*-O.*\|\s*(sh|bash|python)', "Piping download directly to interpreter"),
]


def _validate_command_safety(command: str) -> None:
    """
    Check command for dangerous patterns that could cause system damage.

    Raises:
        Exception: If a dangerous pattern is detected.
    """
    for pattern, description in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            raise Exception(
                f"Dangerous command pattern detected: {description}\n"
                f"Pattern: {pattern}\n"
                f"Command: {command}\n\n"
                f"This command could cause system damage. "
                f"If you're sure this is safe, modify the pattern list in bash.py."
            )


def _kill_process_tree(pid: int) -> None:
    """Kill a process and all its children."""
    try:
        import subprocess
        # Use kill -TERM first, then KILL
        subprocess.run(["kill", "-TERM", f"-{pid}"], capture_output=True)
    except Exception:
        pass
    try:
        os.kill(pid, _signal.SIGTERM)
    except Exception:
        pass


class BashTool:
    """
    AgentTool for executing bash commands.
    Returns stdout+stderr combined, with tail truncation.
    """

    name = "bash"
    label = "bash"
    description = (
        f"Execute a bash command in the current working directory. "
        f"Returns stdout and stderr. Output is truncated to last "
        f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
        f"(whichever is hit first). Optionally provide a timeout in seconds."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Bash command to execute"},
            "timeout": {"type": "number", "description": "Timeout in seconds (optional)"},
        },
        "required": ["command"],
    }

    def __init__(self, cwd: str, command_prefix: Optional[str] = None):
        self.cwd = cwd
        self.command_prefix = command_prefix

    async def execute(
        self,
        tool_call_id: str,
        args: Dict[str, Any],
        signal: Optional[asyncio.Event] = None,
        on_update: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        command: str = args["command"]
        timeout: Optional[float] = args.get("timeout")

        # Validate command safety before execution
        _validate_command_safety(command)

        if self.command_prefix:
            command = f"{self.command_prefix}\n{command}"

        if signal and signal.is_set():
            raise Exception("Operation aborted")

        if not os.path.isdir(self.cwd):
            raise Exception(f"Working directory does not exist: {self.cwd}")

        # We'll collect output chunks and optionally write to a temp file
        chunks: List[bytes] = []
        chunks_bytes = 0
        max_chunks_bytes = DEFAULT_MAX_BYTES * 2
        total_bytes = 0
        temp_file_path: Optional[str] = None
        temp_file = None

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=self.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as e:
            raise Exception(f"Failed to spawn shell: {e}")

        aborted = False

        async def read_output() -> None:
            nonlocal total_bytes, temp_file, temp_file_path, chunks_bytes, aborted

            assert proc.stdout is not None

            while True:
                data = await proc.stdout.read(4096)
                if not data:
                    break

                total_bytes += len(data)

                # Start writing to temp file once we exceed threshold
                if total_bytes > DEFAULT_MAX_BYTES and temp_file is None:
                    fd, temp_file_path = tempfile.mkstemp(prefix="arc-bash-", suffix=".log")
                    temp_file = os.fdopen(fd, "wb")
                    for c in chunks:
                        temp_file.write(c)

                if temp_file is not None:
                    temp_file.write(data)

                # Keep rolling buffer
                chunks.append(data)
                chunks_bytes += len(data)
                while chunks_bytes > max_chunks_bytes and len(chunks) > 1:
                    removed = chunks.pop(0)
                    chunks_bytes -= len(removed)

                # Stream partial output
                if on_update:
                    combined = b"".join(chunks).decode("utf-8", errors="replace")
                    truncation = truncate_tail(combined)
                    on_update({
                        "content": [{"type": "text", "text": truncation.content or ""}],
                        "details": {
                            "truncation": truncation if truncation.truncated else None,
                            "fullOutputPath": temp_file_path,
                        },
                    })

        async def wait_for_abort() -> None:
            nonlocal aborted
            if signal:
                while not signal.is_set():
                    await asyncio.sleep(0.05)
                aborted = True
                if proc.returncode is None:
                    try:
                        if proc.pid:
                            _kill_process_tree(proc.pid)
                            proc.kill()
                    except Exception:
                        pass

        tasks = [asyncio.create_task(read_output())]
        if signal:
            tasks.append(asyncio.create_task(wait_for_abort()))

        timed_out = False
        try:
            if timeout and timeout > 0:
                await asyncio.wait_for(
                    asyncio.gather(tasks[0]),
                    timeout=timeout,
                )
            else:
                await tasks[0]
        except asyncio.TimeoutError:
            timed_out = True
            if proc.pid:
                try:
                    _kill_process_tree(proc.pid)
                    proc.kill()
                except Exception:
                    pass
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if temp_file is not None:
                temp_file.close()

        try:
            exit_code = await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            exit_code = None

        full_output = b"".join(chunks).decode("utf-8", errors="replace")
        truncation = truncate_tail(full_output)
        output_text = truncation.content or "(no output)"

        details: Optional[Dict[str, Any]] = None

        if truncation.truncated:
            details = {
                "truncation": truncation,
                "fullOutputPath": temp_file_path,
            }
            start_line = truncation.total_lines - truncation.output_lines + 1
            end_line = truncation.total_lines

            if truncation.last_line_partial:
                last_line = full_output.split("\n")[-1] if "\n" in full_output else full_output
                last_line_size = format_size(len(last_line.encode("utf-8")))
                output_text += (
                    f"\n\n[Showing last {format_size(truncation.output_bytes)} of line {end_line} "
                    f"(line is {last_line_size}). Full output: {temp_file_path}]"
                )
            elif truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}. "
                    f"Full output: {temp_file_path}]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {temp_file_path}]"
                )

        if aborted:
            if output_text:
                output_text += "\n\n"
            output_text += "Command aborted"
            raise Exception(output_text)

        if timed_out:
            if output_text:
                output_text += "\n\n"
            output_text += f"Command timed out after {timeout} seconds"
            raise Exception(output_text)

        if exit_code is not None and exit_code != 0:
            output_text += f"\n\nCommand exited with code {exit_code}"
            raise Exception(output_text)

        return {
            "content": [{"type": "text", "text": output_text}],
            "details": details,
        }


def create_bash_tool(cwd: str, command_prefix: Optional[str] = None) -> BashTool:
    return BashTool(cwd, command_prefix)
