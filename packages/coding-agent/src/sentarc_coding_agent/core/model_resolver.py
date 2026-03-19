"""Resolve model from provider+id."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def parse_model_spec(model_spec: str) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Parse a model spec string.
    Formats:
    - "model-id"
    - "provider/model-id"
    - "model-id:thinking"
    - "provider/model-id:thinking"
    Returns (provider, model_id, thinking_level).
    """
    thinking: Optional[str] = None
    if ":" in model_spec.split("/")[-1]:
        parts = model_spec.rsplit(":", 1)
        model_spec = parts[0]
        thinking = parts[1]

    if "/" in model_spec:
        slash_idx = model_spec.index("/")
        provider = model_spec[:slash_idx]
        model_id = model_spec[slash_idx + 1:]
        return provider, model_id, thinking

    return None, model_spec, thinking


def resolve_model(
    provider: Optional[str],
    model: Optional[str],
    default_provider: str = "google",
    default_model: str = "gemini-2.5-flash",
) -> Tuple[str, str, Optional[str]]:
    """
    Resolve the final provider, model_id, and thinking level.
    Returns (provider, model_id, thinking_level).
    """
    if not model:
        return default_provider, default_model, None

    parsed_provider, model_id, thinking = parse_model_spec(model)

    # If provider specified in model spec, use it
    resolved_provider = parsed_provider or provider or default_provider

    return resolved_provider, model_id, thinking
