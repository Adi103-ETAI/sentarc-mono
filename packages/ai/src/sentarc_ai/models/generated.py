"""
Generated model definitions.
"""
from __future__ import annotations

PROVIDER_DEFAULTS: dict = {"anthropic":"claude-sonnet-4-6","openai":"gpt-4o","google":"gemini-2.5-flash","ollama":"llama3.2"}


def _make_models():
    from ..types import ModelDef
    return [
        ModelDef(id="claude-opus-4-6",          name="Claude Opus 4.6",   provider="anthropic",api="anthropic-messages", context_window=200_000,max_output=8_192, supports_vision=True,supports_thinking=True, input_cost_per_mtok=15.0,output_cost_per_mtok=75.0, cache_read_cost_per_mtok=1.5,cache_write_cost_per_mtok=18.75),
        ModelDef(id="claude-sonnet-4-6",         name="Claude Sonnet 4.6",provider="anthropic",api="anthropic-messages", context_window=200_000,max_output=8_192, supports_vision=True,supports_thinking=True, input_cost_per_mtok=3.0, output_cost_per_mtok=15.0, cache_read_cost_per_mtok=0.3, cache_write_cost_per_mtok=3.75),
        ModelDef(id="claude-haiku-4-5-20251001", name="Claude Haiku 4.5", provider="anthropic",api="anthropic-messages", context_window=200_000,max_output=8_192, supports_vision=True, input_cost_per_mtok=0.8, output_cost_per_mtok=4.0,  cache_read_cost_per_mtok=0.08,cache_write_cost_per_mtok=1.0),
        ModelDef(id="gpt-4o",                    name="GPT-4o",           provider="openai",   api="openai-completions",context_window=128_000,max_output=16_384,supports_vision=True, input_cost_per_mtok=2.5, output_cost_per_mtok=10.0),
        ModelDef(id="gpt-4o-mini",               name="GPT-4o Mini",      provider="openai",   api="openai-completions",context_window=128_000,max_output=16_384,supports_vision=True, input_cost_per_mtok=0.15,output_cost_per_mtok=0.6),
        ModelDef(id="o3",                        name="o3",               provider="openai",   api="openai-responses",  context_window=200_000,max_output=100_000,supports_vision=True,supports_thinking=True, input_cost_per_mtok=10.0,output_cost_per_mtok=40.0),
        ModelDef(id="o4-mini",                   name="o4-mini",          provider="openai",   api="openai-responses",  context_window=200_000,max_output=100_000,supports_vision=True,supports_thinking=True, input_cost_per_mtok=1.1, output_cost_per_mtok=4.4),
        ModelDef(id="gemini-2.5-pro",            name="Gemini 2.5 Pro",   provider="google",   api="google-generative-ai",context_window=1_000_000,max_output=8_192,supports_vision=True,supports_thinking=True,input_cost_per_mtok=1.25,output_cost_per_mtok=10.0),
        ModelDef(id="gemini-2.5-flash",          name="Gemini 2.5 Flash", provider="google",   api="google-generative-ai",context_window=1_000_000,max_output=8_192,supports_vision=True,supports_thinking=True,input_cost_per_mtok=0.15,output_cost_per_mtok=0.6),
        ModelDef(id="llama3.2",                  name="Llama 3.2",        provider="ollama",   api="openai-completions",context_window=128_000,max_output=4_096, base_url="http://localhost:11434/v1"),
        ModelDef(id="qwen2.5-coder",             name="Qwen 2.5 Coder",   provider="ollama",   api="openai-completions",context_window=128_000,max_output=4_096, base_url="http://localhost:11434/v1"),
        ModelDef(id="deepseek-r1",               name="DeepSeek R1",      provider="ollama",   api="openai-completions",context_window=128_000,max_output=8_192, supports_thinking=True,base_url="http://localhost:11434/v1"),
    ]

GENERATED_MODELS: list = []
def _load(): global GENERATED_MODELS; GENERATED_MODELS = _make_models()
_load()
