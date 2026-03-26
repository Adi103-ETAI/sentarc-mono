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
    def test_absolute_path_inside_cwd(self, tmp_path):
        test_file = tmp_path / "absolute.txt"
        test_file.write_text("ok")
        result = resolve_to_cwd(str(test_file), str(tmp_path))
        assert result == str(test_file)

    def test_absolute_path_outside_cwd_raises(self, tmp_path):
        outside = tmp_path.parent / "outside.txt"
        outside.write_text("nope")
        with pytest.raises(Exception, match="Path traversal detected"):
            resolve_to_cwd(str(outside), str(tmp_path))

    def test_relative_path(self):
        result = resolve_to_cwd("relative/file.txt", "/cwd")
        assert result == "/cwd/relative/file.txt"

    def test_tilde_outside_cwd_raises(self):
        with pytest.raises(Exception, match="Path traversal detected"):
            resolve_to_cwd("~/file.txt", "/cwd")

    def test_dot_path(self):
        result = resolve_to_cwd(".", "/cwd")
        assert result == "/cwd"

    def test_parent_traversal_raises(self, tmp_path):
        with pytest.raises(Exception, match="Path traversal detected"):
            resolve_to_cwd("../outside.txt", str(tmp_path))

    def test_prefix_sibling_escape_raises(self, tmp_path):
        root = tmp_path / "xyz-root"
        root.mkdir()
        sibling = tmp_path / "xyz-root-alt"
        sibling.mkdir()
        with pytest.raises(Exception, match="Path traversal detected"):
            resolve_to_cwd("../xyz-root-alt/file.txt", str(root))


class TestResolveReadPath:
    def test_existing_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        result = resolve_read_path("test.txt", str(tmp_path))
        assert result == str(test_file)

    def test_nonexistent_file_returns_resolved(self):
        result = resolve_read_path("nonexistent.txt", "/cwd")
        assert result == "/cwd/nonexistent.txt"

    def test_absolute_existing_file_outside_cwd_raises(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        with pytest.raises(Exception, match="Path traversal detected"):
            resolve_read_path(str(test_file), "/any/cwd")

    def test_symlink_escape_outside_cwd_raises(self, tmp_path):
        cwd = tmp_path / "root"
        cwd.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "secret.txt"
        target.write_text("secret")

        link = cwd / "escape-link"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("Symlink not supported in this environment")

        with pytest.raises(Exception, match="Path traversal detected"):
            resolve_read_path("escape-link", str(cwd))

    def test_symlink_inside_cwd_is_allowed(self, tmp_path):
        cwd = tmp_path / "root"
        cwd.mkdir()
        target = cwd / "safe.txt"
        target.write_text("ok")

        link = cwd / "safe-link"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("Symlink not supported in this environment")

        resolved = resolve_read_path("safe-link", str(cwd))
        assert resolved == str(target)
