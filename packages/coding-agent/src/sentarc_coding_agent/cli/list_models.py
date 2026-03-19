"""Print available models."""

from __future__ import annotations
from typing import Optional


def list_models(search: Optional[str] = None) -> None:
    """Print available models, optionally filtered by search pattern."""
    try:
        from sentarc_ai.models import list_models as ai_list_models
        models = ai_list_models()
    except Exception:
        models = []

    if search:
        search_lower = search.lower()
        models = [m for m in models if search_lower in str(m).lower()]

    if not models:
        print("No models found.")
        return

    for model in models:
        if hasattr(model, "provider") and hasattr(model, "id"):
            print(f"{model.provider}/{model.id}")
        else:
            print(str(model))
