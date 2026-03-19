"""Config validation helpers."""
from __future__ import annotations

from typing import Optional


def validate_config(provider: Optional[str], model_id: Optional[str]) -> bool:
    """Validate provider+model combination. Returns True if valid or defaults used."""
    if not provider and not model_id:
        return True
    try:
        from sentarc_coding_agent.core.model_resolver import resolve_model
        resolve_model(provider, model_id)
        return True
    except Exception as e:
        print(f"Invalid model configuration: {e}")
        return False
