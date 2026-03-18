#!/usr/bin/env python3
"""
Sync versions across all packages in sentarc-mono.
Ensures lockstep versioning across the monorepo.

Usage:
    python scripts/sync-versions.py [--set VERSION]
"""

import sys
import re
from pathlib import Path

PACKAGES_DIR = Path(__file__).parent.parent / "packages"
VERSION_PATTERN = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


def read_version(pyproject_path: Path) -> str | None:
    """Read version from pyproject.toml."""
    content = pyproject_path.read_text()
    match = VERSION_PATTERN.search(content)
    return match.group(1) if match else None


def write_version(pyproject_path: Path, new_version: str) -> None:
    """Write version to pyproject.toml."""
    content = pyproject_path.read_text()
    updated = VERSION_PATTERN.sub(f'version = "{new_version}"', content, count=1)
    pyproject_path.write_text(updated)


def get_all_packages() -> dict[str, Path]:
    """Get all packages with their pyproject.toml paths."""
    packages = {}
    for pkg_dir in PACKAGES_DIR.iterdir():
        if pkg_dir.is_dir():
            pyproject = pkg_dir / "pyproject.toml"
            if pyproject.exists():
                packages[pkg_dir.name] = pyproject
    return packages


def bump_version(version: str, bump_type: str) -> str:
    """Bump version string."""
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {version}")

    major, minor, patch = map(int, parts)

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")


def main():
    packages = get_all_packages()

    if not packages:
        print("No packages found!")
        sys.exit(1)

    # Read all versions
    versions = {}
    for name, pyproject in packages.items():
        version = read_version(pyproject)
        if version:
            versions[name] = version
            print(f"  {name}: {version}")

    # Check for --set or --bump argument
    new_version = None
    if len(sys.argv) >= 3:
        if sys.argv[1] == "--set":
            new_version = sys.argv[2]
        elif sys.argv[1] == "--bump":
            bump_type = sys.argv[2]
            if bump_type not in ("major", "minor", "patch"):
                print(f"Invalid bump type: {bump_type}")
                print("Valid types: major, minor, patch")
                sys.exit(1)
            # Use first package version as base
            base_version = list(versions.values())[0]
            new_version = bump_version(base_version, bump_type)

    if new_version:
        print(f"\nSetting all packages to version: {new_version}")
        for name, pyproject in packages.items():
            write_version(pyproject, new_version)
            print(f"  Updated {name}")
        print(f"\n✅ All packages updated to {new_version}")
    else:
        # Verify all versions match
        unique_versions = set(versions.values())
        if len(unique_versions) > 1:
            print("\n❌ ERROR: Not all packages have the same version!")
            print("Run: python scripts/sync-versions.py --set <version>")
            sys.exit(1)
        else:
            print("\n✅ All packages at same version (lockstep)")


if __name__ == "__main__":
    main()
