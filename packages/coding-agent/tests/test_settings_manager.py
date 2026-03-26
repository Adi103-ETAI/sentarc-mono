import json

from sentarc_coding_agent.core.settings_manager import load_settings


def test_load_settings_merges_project_over_global(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    global_dir = home / ".arc" / "agent"
    project_dir = project / ".arc"
    global_dir.mkdir(parents=True)
    project_dir.mkdir(parents=True)

    (global_dir / "settings.json").write_text(
        json.dumps(
            {
                "provider": "openai",
                "model": "gpt-4o",
                "thinking": "high",
                "tools": ["read", "bash"],
                "bashSecurityProfile": "standard",
                "eventLogEnabled": False,
            }
        ),
        encoding="utf-8",
    )

    (project_dir / "settings.json").write_text(
        json.dumps(
            {
                "provider": "google",
                "bashSecurityProfile": "read-only",
                "eventLogEnabled": True,
                "eventLogPath": "/tmp/project-events.jsonl",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("HOME", str(home))

    settings = load_settings(str(project))

    assert settings.provider == "google"
    assert settings.model == "gpt-4o"
    assert settings.thinking == "high"
    assert settings.tools == ["read", "bash"]
    assert settings.bash_security_profile == "read-only"
    assert settings.event_log_enabled is True
    assert settings.event_log_path == "/tmp/project-events.jsonl"


def test_load_settings_supports_quiet_startup_key_variants(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    global_dir = home / ".arc" / "agent"
    project_dir = project / ".arc"
    global_dir.mkdir(parents=True)
    project_dir.mkdir(parents=True)

    (global_dir / "settings.json").write_text(
        json.dumps({"quietStartup": True}),
        encoding="utf-8",
    )
    (project_dir / "settings.json").write_text(
        json.dumps({"quiet_startup": False}),
        encoding="utf-8",
    )

    monkeypatch.setenv("HOME", str(home))

    settings = load_settings(str(project))

    assert settings.quiet_startup is False
