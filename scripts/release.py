#!/usr/bin/env python3
"""
Release script for sentarc-mono.

Usage: python scripts/release.py <major|minor|patch>

Steps:
1. Check for uncommitted changes
2. Bump version across all packages
3. Update CHANGELOG.md files: [Unreleased] -> [version] - date
4. Commit and tag
5. (Optional) Publish to PyPI
6. Add new [Unreleased] section to changelogs
7. Commit and push
"""

import subprocess
import sys
import re
from datetime import date
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
PACKAGES_DIR = ROOT_DIR / "packages"


def run(cmd: str, check: bool = True, capture: bool = False) -> str | None:
    """Run a shell command."""
    print(f"$ {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=ROOT_DIR,
        capture_output=capture,
        text=True,
    )
    if check and result.returncode != 0:
        print(f"Command failed: {cmd}")
        sys.exit(1)
    return result.stdout if capture else None


def get_version() -> str:
    """Get current version from first package."""
    pyproject = PACKAGES_DIR / "ai" / "pyproject.toml"
    content = pyproject.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not match:
        print("Could not read version!")
        sys.exit(1)
    return match.group(1)


def get_changelogs() -> list[Path]:
    """Get all CHANGELOG.md files."""
    changelogs = []
    for pkg_dir in PACKAGES_DIR.iterdir():
        if pkg_dir.is_dir():
            changelog = pkg_dir / "CHANGELOG.md"
            if changelog.exists():
                changelogs.append(changelog)
    # Also check root
    root_changelog = ROOT_DIR / "CHANGELOG.md"
    if root_changelog.exists():
        changelogs.append(root_changelog)
    return changelogs


def update_changelogs_for_release(version: str) -> None:
    """Update [Unreleased] sections with version and date."""
    today = date.today().isoformat()
    changelogs = get_changelogs()

    for changelog in changelogs:
        content = changelog.read_text()
        if "## [Unreleased]" not in content:
            print(f"  Skipping {changelog}: no [Unreleased] section")
            continue

        updated = content.replace(
            "## [Unreleased]",
            f"## [{version}] - {today}"
        )
        changelog.write_text(updated)
        print(f"  Updated {changelog}")


def add_unreleased_section() -> None:
    """Add [Unreleased] section to all changelogs."""
    changelogs = get_changelogs()
    unreleased = "## [Unreleased]\n\n"

    for changelog in changelogs:
        content = changelog.read_text()
        # Insert after "# Changelog\n\n"
        updated = re.sub(
            r"^(# Changelog\n\n)",
            f"\\1{unreleased}",
            content,
        )
        changelog.write_text(updated)
        print(f"  Added [Unreleased] to {changelog}")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("major", "minor", "patch"):
        print("Usage: python scripts/release.py <major|minor|patch>")
        sys.exit(1)

    bump_type = sys.argv[1]

    print("\n=== Sentarc Release Script ===\n")

    # 1. Check for uncommitted changes
    print("Checking for uncommitted changes...")
    status = run("git status --porcelain", capture=True)
    if status and status.strip():
        print("Error: Uncommitted changes detected. Commit or stash first.")
        print(status)
        sys.exit(1)
    print("  Working directory clean\n")

    # 2. Bump version
    print(f"Bumping version ({bump_type})...")
    run(f"python scripts/sync-versions.py --bump {bump_type}")
    version = get_version()
    print(f"  New version: {version}\n")

    # 3. Update changelogs
    print("Updating CHANGELOG.md files...")
    update_changelogs_for_release(version)
    print()

    # 4. Commit and tag
    print("Committing and tagging...")
    run("git add .")
    run(f'git commit -m "Release v{version}"')
    run(f"git tag v{version}")
    print()

    # 5. Optional: Publish to PyPI
    print("To publish to PyPI, run:")
    print("  pip install build twine")
    print("  cd packages/ai && python -m build && twine upload dist/*")
    print("  cd packages/agent && python -m build && twine upload dist/*")
    print("  cd packages/tui && python -m build && twine upload dist/*")
    print("  cd packages/coding-agent && python -m build && twine upload dist/*")
    print()

    # 6. Add [Unreleased] sections
    print("Adding [Unreleased] sections for next cycle...")
    add_unreleased_section()
    print()

    # 7. Commit
    print("Committing changelog updates...")
    run("git add .")
    run('git commit -m "Add [Unreleased] section for next cycle"')
    print()

    # 8. Push
    print("Pushing to remote...")
    run("git push origin main")
    run(f"git push origin v{version}")
    print()

    print(f"=== Released v{version} ===")


if __name__ == "__main__":
    main()
