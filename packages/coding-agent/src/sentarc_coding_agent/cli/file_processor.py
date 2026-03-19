"""Load context files from CLI @file arguments."""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, List


def load_file_args(file_args: List[str], cwd: str) -> List[Dict[str, Any]]:
    """
    Load context files from CLI @file arguments.
    Returns list of dicts with 'path' and 'content' (or 'data'/'mime_type' for images).
    """
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    results: List[Dict[str, Any]] = []

    for raw_path in file_args:
        # Expand ~ and resolve relative paths
        if raw_path.startswith("~/"):
            abs_path = os.path.expanduser(raw_path)
        elif os.path.isabs(raw_path):
            abs_path = raw_path
        else:
            abs_path = os.path.join(cwd, raw_path)

        if not os.path.exists(abs_path):
            print(f"Warning: File not found: {abs_path}")
            continue

        ext = os.path.splitext(abs_path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            try:
                with open(abs_path, "rb") as f:
                    data = base64.b64encode(f.read()).decode("ascii")
                mime_map = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                }
                results.append({
                    "path": raw_path,
                    "type": "image",
                    "data": data,
                    "mime_type": mime_map.get(ext, "image/jpeg"),
                })
            except Exception as e:
                print(f"Warning: Could not read image {abs_path}: {e}")
        else:
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                results.append({"path": raw_path, "type": "text", "content": content})
            except Exception as e:
                print(f"Warning: Could not read file {abs_path}: {e}")

    return results
