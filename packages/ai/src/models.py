"""
Model registry.
Python port of packages/ai/src/models.generated.ts from sentarc-mono.

In sentarc-mono this file is auto-generated from an external API.
Here we define the most common models statically.
Add your own via register_model() or ~/.sentarc/agent/models.json (loaded by sentarc_agent).
"""
from __future__ import annotations
from typing import Optional
from .types import ModelDef

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, ModelDef] = {}


def register_model(model: ModelDef) -> None:
    _REGISTRY[f"{model.provider}/{model.id}"] = model


def get_model(provider: str, model_id: str) -> Optional[ModelDef]:
    return _REGISTRY.get(f"{provider}/{model_id}")


def list_models(provider: Optional[str] = None) -> list[ModelDef]:
    if provider:
        return [m for m in _REGISTRY.values() if m.provider == provider]
    return list(_REGISTRY.values())


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

for _m in [
    ModelDef(
        id="claude-opus-4-6",
        provider="anthropic", api="anthropic",
        context_window=200_000, max_output=8_192,
        supports_thinking=True,
        input_cost_per_mtok=15.0, output_cost_per_mtok=75.0,
    ),
    ModelDef(
        id="claude-sonnet-4-6",
        provider="anthropic", api="anthropic",
        context_window=200_000, max_output=8_192,
        supports_thinking=True,
        input_cost_per_mtok=3.0, output_cost_per_mtok=15.0,
    ),
    ModelDef(
        id="claude-haiku-4-5-20251001",
        provider="anthropic", api="anthropic",
        context_window=200_000, max_output=8_192,
        input_cost_per_mtok=0.8, output_cost_per_mtok=4.0,
    ),
]:
    register_model(_m)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

for _m in [
    ModelDef(
        id="gpt-4o",
        provider="openai", api="openai",
        context_window=128_000, max_output=16_384,
        input_cost_per_mtok=2.5, output_cost_per_mtok=10.0,
    ),
    ModelDef(
        id="gpt-4o-mini",
        provider="openai", api="openai",
        context_window=128_000, max_output=16_384,
        input_cost_per_mtok=0.15, output_cost_per_mtok=0.6,
    ),
    ModelDef(
        id="o3",
        provider="openai", api="openai",
        context_window=200_000, max_output=100_000,
        supports_thinking=True,
        input_cost_per_mtok=10.0, output_cost_per_mtok=40.0,
    ),
    ModelDef(
        id="o4-mini",
        provider="openai", api="openai",
        context_window=200_000, max_output=100_000,
        supports_thinking=True,
        input_cost_per_mtok=1.1, output_cost_per_mtok=4.4,
    ),
]:
    register_model(_m)


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------

for _m in [
    ModelDef(
        id="gemini-2.5-pro",
        provider="google", api="google",
        context_window=1_000_000, max_output=8_192,
        supports_thinking=True,
        input_cost_per_mtok=1.25, output_cost_per_mtok=10.0,
    ),
    ModelDef(
        id="gemini-2.5-flash",
        provider="google", api="google",
        context_window=1_000_000, max_output=8_192,
        supports_thinking=True,
        input_cost_per_mtok=0.15, output_cost_per_mtok=0.6,
    ),
]:
    register_model(_m)


# ---------------------------------------------------------------------------
# Ollama (local — no cost)
# ---------------------------------------------------------------------------

for _m in [
    ModelDef(
        id="llama3.2",
        provider="ollama", api="openai",        # Ollama speaks OpenAI API
        context_window=128_000, max_output=4_096,
        base_url="http://localhost:11434/v1",
    ),
    ModelDef(
        id="qwen2.5-coder",
        provider="ollama", api="openai",
        context_window=128_000, max_output=4_096,
        base_url="http://localhost:11434/v1",
    ),
]:
    register_model(_m)


# ---------------------------------------------------------------------------
# Provider defaults  (used when only a provider name is given)
# ---------------------------------------------------------------------------

PROVIDER_DEFAULTS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai":    "gpt-4o",
    "google":    "gemini-2.5-flash",
    "ollama":    "llama3.2",
}


def resolve_model(provider: str, model_id: Optional[str] = None) -> ModelDef:
    """
    Resolve a ModelDef from provider + optional model id.
    Falls back to the provider default if model_id is not given.
    Raises ValueError if nothing matches.
    """
    mid = model_id or PROVIDER_DEFAULTS.get(provider)
    if not mid:
        raise ValueError(f"Unknown provider: {provider!r}")
    m = get_model(provider, mid)
    if not m:
        # Unknown model — build a minimal ModelDef on the fly
        api = "openai" if provider in ("ollama",) else provider
        return ModelDef(id=mid, provider=provider, api=api)
    return m
