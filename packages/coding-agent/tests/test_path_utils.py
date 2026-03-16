"""Tests for expand_path and resolve_to_cwd."""

import os
import pytest
from pathlib import Path

from sentarc_coding_agent.core.tools.path_utils import (
    expand_path,
    resolve_to_cwd,
    resolve_read_path,
)


class TestExpandPath:
    def test_tilde_only(self):
        result = expand_path("~")
        assert result == str(Path.home())

    def test_tilde_path(self):
        result = expand_path("~/foo/bar")
        assert result == str(Path.home() / "foo" / "bar")

    def test_absolute_path(self):
        result = expand_path("/absolute/path")
        assert result == "/absolute/path"

    def test_relative_path(self):
        result = expand_path("relative/path")
        assert result == "relative/path"

    def test_at_prefix_stripped(self):
        result = expand_path("@/some/path")
        assert result == "/some/path"

    def test_unicode_spaces_normalized(self):
        # Non-breaking space should become regular space
        result = expand_path("path\u00A0with\u00A0spaces")
        assert "\u00A0" not in result
        assert " " in result


class TestResolveToCwd:
    def test_absolute_path(self):
        result = resolve_to_cwd("/absolute/path", "/cwd")
        assert result == "/absolute/path"

    def test_relative_path(self):
        result = resolve_to_cwd("relative/file.txt", "/cwd")
        assert result == "/cwd/relative/file.txt"

    def test_tilde_expansion(self):
        result = resolve_to_cwd("~/file.txt", "/cwd")
        assert result == str(Path.home() / "file.txt")

    def test_dot_path(self):
        result = resolve_to_cwd(".", "/cwd")
        assert result == "/cwd"


class TestResolveReadPath:
    def test_existing_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        result = resolve_read_path("test.txt", str(tmp_path))
        assert result == str(test_file)

    def test_nonexistent_file_returns_resolved(self):
        result = resolve_read_path("nonexistent.txt", "/cwd")
        assert result == "/cwd/nonexistent.txt"

    def test_absolute_existing_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        result = resolve_read_path(str(test_file), "/any/cwd")
        assert result == str(test_file)
