"""
Tests for sentarc_tui.utils — visible_width, strip_ansi, truncate_to_width.
"""
import pytest
from sentarc_tui.utils import strip_ansi, visible_width, truncate_to_width


# ---------------------------------------------------------------------------
# strip_ansi
# ---------------------------------------------------------------------------

class TestStripAnsi:
    def test_plain_text_unchanged(self):
        assert strip_ansi("hello world") == "hello world"

    def test_removes_colour_codes(self):
        assert strip_ansi("\x1b[31mred\x1b[0m") == "red"

    def test_removes_bold(self):
        assert strip_ansi("\x1b[1mbold\x1b[22m") == "bold"

    def test_removes_256_colour(self):
        assert strip_ansi("\x1b[38;5;200mcolour\x1b[0m") == "colour"

    def test_removes_rgb_colour(self):
        assert strip_ansi("\x1b[38;2;255;128;0mRGB\x1b[0m") == "RGB"

    def test_removes_cursor_movement(self):
        assert strip_ansi("\x1b[2Aup\x1b[2B") == "up"

    def test_empty_string(self):
        assert strip_ansi("") == ""

    def test_multiple_sequences(self):
        text = "\x1b[1m\x1b[32mhello\x1b[0m world\x1b[0m"
        assert strip_ansi(text) == "hello world"


# ---------------------------------------------------------------------------
# visible_width
# ---------------------------------------------------------------------------

class TestVisibleWidth:
    def test_ascii(self):
        assert visible_width("hello") == 5

    def test_empty(self):
        assert visible_width("") == 0

    def test_strips_ansi_before_counting(self):
        assert visible_width("\x1b[31mred\x1b[0m") == 3

    def test_cjk_double_width(self):
        # Each CJK ideograph is 2 columns wide
        assert visible_width("你好") == 4

    def test_mixed_ascii_and_cjk(self):
        assert visible_width("hi你好") == 6  # 2 + 4

    def test_ansi_with_cjk(self):
        assert visible_width("\x1b[1m你\x1b[0m好") == 4


# ---------------------------------------------------------------------------
# truncate_to_width
# ---------------------------------------------------------------------------

class TestTruncateToWidth:
    def test_no_truncation_needed(self):
        assert truncate_to_width("hello", 10) == "hello"

    def test_exact_width(self):
        assert truncate_to_width("hello", 5) == "hello"

    def test_truncates_ascii(self):
        result = truncate_to_width("hello world", 5)
        assert result == "hello"
        assert visible_width(result) <= 5

    def test_zero_max_width(self):
        assert truncate_to_width("hello", 0) == ""

    def test_preserves_ansi(self):
        coloured = "\x1b[31mhello\x1b[0m"
        result = truncate_to_width(coloured, 3)
        # Visible part is "hel", ANSI codes must still be present
        assert strip_ansi(result) == "hel"

    def test_cjk_no_half_character(self):
        # "你好世界" = 8 columns; truncating to 3 should give "你" (2 cols)
        # because "好" would push it to 4 which exceeds 3
        result = truncate_to_width("你好世界", 3)
        w = visible_width(result)
        assert w <= 3
        assert result == "你"

    def test_empty_input(self):
        assert truncate_to_width("", 10) == ""
