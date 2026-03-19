"""Load custom model definitions from ~/.arc/agent/models.json."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from sentarc_coding_agent.config import get_models_path


def load_custom_models() -> List[Dict[str, Any]]:
    """Load custom model definitions from models.json."""
    path = get_models_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def get_all_models() -> List[Dict[str, Any]]:
    """Get all available models (built-in + custom)."""
    models = []

    # Try to get built-in models from sentarc_ai
    try:
        from sentarc_ai.models import list_models
        for m in list_models():
            if hasattr(m, "provider") and hasattr(m, "id"):
                models.append({"provider": m.provider, "id": m.id, "name": getattr(m, "name", m.id)})
            elif isinstance(m, dict):
                models.append(m)
    except Exception:
        pass

    # Add custom models
    models.extend(load_custom_models())
    return models
