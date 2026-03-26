import pytest
import asyncio
import time

from sentarc_coding_agent.core.tools.bash import BashTool


@pytest.mark.asyncio
async def test_bash_standard_allows_read_command(tmp_path):
    tool = BashTool(cwd=str(tmp_path), security_profile="standard")
    result = await tool.execute("tc1", {"command": "printf hello"})
    assert "hello" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_bash_read_only_blocks_write_like_command(tmp_path):
    tool = BashTool(cwd=str(tmp_path), security_profile="read-only")
    with pytest.raises(Exception, match="potentially dangerous pattern"):
        await tool.execute("tc2", {"command": "touch new_file.txt"})


@pytest.mark.asyncio
async def test_bash_custom_block_pattern_applies(tmp_path):
    tool = BashTool(
        cwd=str(tmp_path),
        security_profile="standard",
        blocked_patterns=[r"\becho\b"],
    )
    with pytest.raises(Exception, match="potentially dangerous pattern"):
        await tool.execute("tc3", {"command": "echo hello"})


@pytest.mark.asyncio
async def test_bash_abort_is_fast_with_escalation(tmp_path):
    tool = BashTool(cwd=str(tmp_path), security_profile="standard")
    abort_signal = asyncio.Event()

    async def _trigger_abort():
        await asyncio.sleep(0.05)
        abort_signal.set()

    start = time.monotonic()
    trigger_task = asyncio.create_task(_trigger_abort())
    try:
        with pytest.raises(Exception, match="Command aborted"):
            await tool.execute("tc4", {"command": "sleep 10"}, signal=abort_signal)
    finally:
        await trigger_task

    elapsed = time.monotonic() - start
    assert elapsed < 2.0
