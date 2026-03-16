"""YAML frontmatter parsing."""

from __future__ import annotations

from typing import Any, Dict


def parse_frontmatter(content: str) -> Dict[str, Any]:
    """
    Parse YAML frontmatter from a markdown file.
    Returns the frontmatter dict (empty if none present).
    """
    if not content.startswith("---"):
        return {}

    end_idx = content.find("---", 3)
    if end_idx == -1:
        return {}

    fm_text = content[3:end_idx].strip()
    if not fm_text:
        return {}

    try:
        import yaml
        result = yaml.safe_load(fm_text)
        return result if isinstance(result, dict) else {}
    except Exception:
        pass

    # Fallback: manual key:value parsing
    result: Dict[str, Any] = {}
    for line in fm_text.splitlines():
        line = line.strip()
        if ":" in line and not line.startswith("#"):
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            # Remove quotes
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            # Convert booleans
            if value.lower() == "true":
                result[key] = True
            elif value.lower() == "false":
                result[key] = False
            else:
                result[key] = value
    return result
