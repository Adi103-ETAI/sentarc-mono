"""Tests for truncate_head and truncate_tail."""

import pytest
from sentarc_coding_agent.core.tools.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationOptions,
    format_size,
    truncate_head,
    truncate_tail,
    truncate_line,
    GREP_MAX_LINE_LENGTH,
)


class TestFormatSize:
    def test_bytes(self):
        assert format_size(100) == "100B"

    def test_kb(self):
        assert format_size(1024) == "1.0KB"
        assert format_size(2048) == "2.0KB"

    def test_mb(self):
        assert format_size(1024 * 1024) == "1.0MB"


class TestTruncateHead:
    def test_no_truncation_needed(self):
        content = "line1\nline2\nline3"
        result = truncate_head(content)
        assert result.truncated is False
        assert result.content == content
        assert result.truncated_by is None
        assert result.total_lines == 3

    def test_truncation_by_lines(self):
        lines = [f"line{i}" for i in range(100)]
        content = "\n".join(lines)
        result = truncate_head(content, TruncationOptions(max_lines=10))
        assert result.truncated is True
        assert result.truncated_by == "lines"
        assert result.output_lines == 10
        assert result.content == "\n".join(lines[:10])

    def test_truncation_by_bytes(self):
        # Create content that exceeds byte limit
        content = "x" * (DEFAULT_MAX_BYTES + 1000)
        result = truncate_head(content, TruncationOptions(max_bytes=100))
        assert result.truncated is True
        assert result.truncated_by == "bytes"

    def test_first_line_exceeds_limit(self):
        content = "x" * 1000 + "\nline2"
        result = truncate_head(content, TruncationOptions(max_bytes=500))
        assert result.first_line_exceeds_limit is True
        assert result.content == ""

    def test_empty_content(self):
        result = truncate_head("")
        assert result.truncated is False
        assert result.total_lines == 1

    def test_preserves_complete_lines(self):
        content = "abc\ndef\nghi"
        result = truncate_head(content, TruncationOptions(max_lines=2))
        assert result.truncated is True
        assert result.content == "abc\ndef"
        assert "\n" not in result.content.rstrip("\n") or result.content.count("\n") == 1


class TestTruncateTail:
    def test_no_truncation_needed(self):
        content = "line1\nline2\nline3"
        result = truncate_tail(content)
        assert result.truncated is False
        assert result.content == content

    def test_truncation_by_lines(self):
        lines = [f"line{i}" for i in range(100)]
        content = "\n".join(lines)
        result = truncate_tail(content, TruncationOptions(max_lines=10))
        assert result.truncated is True
        assert result.truncated_by == "lines"
        assert result.output_lines == 10
        # Should contain the LAST 10 lines
        assert result.content == "\n".join(lines[-10:])

    def test_truncation_by_bytes(self):
        content = "x" * (DEFAULT_MAX_BYTES + 1000)
        result = truncate_tail(content, TruncationOptions(max_bytes=100))
        assert result.truncated is True
        assert result.truncated_by == "bytes"

    def test_keeps_last_lines(self):
        content = "first\nsecond\nthird\nfourth\nfifth"
        result = truncate_tail(content, TruncationOptions(max_lines=2))
        assert result.truncated is True
        assert "fourth" in result.content
        assert "fifth" in result.content
        assert "first" not in result.content

    def test_last_line_partial(self):
        # Single line that exceeds byte limit
        content = "x" * 1000
        result = truncate_tail(content, TruncationOptions(max_bytes=100))
        assert result.truncated is True
        assert result.last_line_partial is True


class TestTruncateLine:
    def test_no_truncation(self):
        line = "short line"
        text, was_truncated = truncate_line(line)
        assert text == line
        assert was_truncated is False

    def test_truncation(self):
        line = "x" * (GREP_MAX_LINE_LENGTH + 100)
        text, was_truncated = truncate_line(line)
        assert was_truncated is True
        assert "[truncated]" in text
        assert len(text) < len(line)

    def test_custom_max_chars(self):
        line = "hello world"
        text, was_truncated = truncate_line(line, max_chars=5)
        assert was_truncated is True
        assert "hello" in text
        assert "[truncated]" in text
