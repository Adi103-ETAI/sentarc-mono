import pytest

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
