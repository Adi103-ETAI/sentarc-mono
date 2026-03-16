"""Tests for build_system_prompt."""

import os
import pytest
from sentarc_coding_agent.core.system_prompt import build_system_prompt


class TestBuildSystemPrompt:
    def test_default_prompt_contains_tools(self):
        prompt = build_system_prompt(cwd="/test/cwd")
        assert "read" in prompt
        assert "bash" in prompt
        assert "edit" in prompt
        assert "write" in prompt

    def test_default_prompt_contains_guidelines(self):
        prompt = build_system_prompt(cwd="/test/cwd")
        assert "Guidelines:" in prompt
        assert "Be concise" in prompt

    def test_default_prompt_contains_cwd(self):
        prompt = build_system_prompt(cwd="/my/project")
        assert "/my/project" in prompt

    def test_default_prompt_contains_date(self):
        from datetime import date
        prompt = build_system_prompt(cwd="/test")
        assert date.today().isoformat() in prompt

    def test_custom_prompt(self):
        prompt = build_system_prompt(
            custom_prompt="You are a custom assistant.",
            cwd="/test",
        )
        assert "You are a custom assistant." in prompt
        assert "custom assistant" in prompt

    def test_append_system_prompt(self):
        prompt = build_system_prompt(
            append_system_prompt="Extra instructions here.",
            cwd="/test",
        )
        assert "Extra instructions here." in prompt

    def test_selected_tools(self):
        prompt = build_system_prompt(
            selected_tools=["read", "grep"],
            cwd="/test",
        )
        assert "read" in prompt
        assert "grep" in prompt

    def test_no_bash_uses_bash_guideline(self):
        prompt = build_system_prompt(
            selected_tools=["bash"],
            cwd="/test",
        )
        assert "bash for file operations" in prompt

    def test_bash_with_grep_ls(self):
        prompt = build_system_prompt(
            selected_tools=["bash", "grep", "find", "ls"],
            cwd="/test",
        )
        assert "Prefer grep/find/ls" in prompt

    def test_context_files_included(self):
        prompt = build_system_prompt(
            cwd="/test",
            context_files=[
                {"path": "README.md", "content": "This is the readme."}
            ],
        )
        assert "README.md" in prompt
        assert "This is the readme." in prompt

    def test_custom_prompt_with_context_files(self):
        prompt = build_system_prompt(
            custom_prompt="Base prompt.",
            cwd="/test",
            context_files=[
                {"path": "notes.md", "content": "Important notes."}
            ],
        )
        assert "Base prompt." in prompt
        assert "Important notes." in prompt

    def test_prompt_guidelines_added(self):
        prompt = build_system_prompt(
            cwd="/test",
            prompt_guidelines=["Always use snake_case.", "Write tests first."],
        )
        assert "Always use snake_case." in prompt
        assert "Write tests first." in prompt
