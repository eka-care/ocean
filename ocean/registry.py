from __future__ import annotations

from ocean.exceptions import ModelNotFoundError
from ocean.models import Modality
from ocean.providers.base import Provider

# ── Model catalog ──
# Each entry: (model_name, provider, list_of_modalities, description)

MODEL_CATALOG: list[tuple[str, str, list[str], str]] = [
    # ━━ OpenAI ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # GPT-5 series (frontier)
    ("gpt-5.4", "openai", ["text", "vision"], "Flagship model — complex reasoning and coding"),
    ("gpt-5.4-pro", "openai", ["text", "vision"], "Maximum performance on hardest tasks"),
    ("gpt-5.4-mini", "openai", ["text", "vision"], "Fast frontier model with computer use"),
    ("gpt-5.4-nano", "openai", ["text", "vision"], "Smallest GPT-5.4 variant"),
    ("gpt-5", "openai", ["text", "vision"], "GPT-5 base model"),
    ("gpt-5-mini", "openai", ["text", "vision"], "Fast and cost-efficient GPT-5"),
    ("gpt-5-nano", "openai", ["text", "vision"], "Lightweight GPT-5"),

    # GPT-4 series (still available)
    ("gpt-4.1", "openai", ["text", "vision"], "GPT-4.1"),
    ("gpt-4.1-mini", "openai", ["text", "vision"], "GPT-4.1 Mini"),
    ("gpt-4.1-nano", "openai", ["text"], "GPT-4.1 Nano — smallest and fastest"),
    ("gpt-4o", "openai", ["text", "vision"], "GPT-4o multimodal"),
    ("gpt-4o-mini", "openai", ["text", "vision"], "Fast and affordable GPT-4o"),

    # Reasoning (o-series)
    ("o3", "openai", ["text", "vision"], "Advanced reasoning model"),
    ("o3-pro", "openai", ["text", "vision"], "Pro-tier reasoning — more compute"),
    ("o3-mini", "openai", ["text"], "Small reasoning model for STEM"),
    ("o4-mini", "openai", ["text", "vision"], "Fast reasoning — coding and visual tasks"),
    ("o1", "openai", ["text"], "Original reasoning model"),
    ("o3-deep-research", "openai", ["text"], "Deep research and analysis"),
    ("o4-mini-deep-research", "openai", ["text"], "Fast deep research"),

    # Codex
    ("gpt-5-codex", "openai", ["text"], "Most capable agentic coding model"),
    ("gpt-5.3-codex", "openai", ["text"], "Agentic coding model"),

    # Image generation
    ("gpt-image-1.5", "openai", ["image_gen"], "Latest image generation"),
    ("gpt-image-1", "openai", ["image_gen"], "Image generation and editing"),
    ("gpt-image-1-mini", "openai", ["image_gen"], "Fast image generation"),
    ("chatgpt-image-latest", "openai", ["image_gen"], "Latest ChatGPT image model"),

    # Audio — TTS
    ("tts-1", "openai", ["tts"], "Text-to-speech"),
    ("tts-1-hd", "openai", ["tts"], "Text-to-speech (high quality)"),
    ("gpt-4o-mini-tts", "openai", ["tts"], "GPT-4o Mini text-to-speech"),

    # Audio — Transcription
    ("whisper-1", "openai", ["transcription"], "Speech-to-text transcription"),
    ("gpt-4o-transcribe", "openai", ["transcription"], "GPT-4o speech-to-text"),
    ("gpt-4o-mini-transcribe", "openai", ["transcription"], "GPT-4o Mini speech-to-text"),

    # Deprecated OpenAI models (kept for reference)
    ("dall-e-3", "openai", ["image_gen"], "Image generation (deprecated)"),
    ("dall-e-2", "openai", ["image_gen"], "Image generation (deprecated)"),

    # ━━ Gemini ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Gemini 3 series
    ("gemini-3.1-pro-preview", "gemini", ["text", "vision", "transcription"], "Most capable Gemini — reasoning and agentic"),
    ("gemini-3-flash-preview", "gemini", ["text", "vision", "transcription"], "Pro-level intelligence at Flash speed"),
    ("gemini-3.1-flash-lite-preview", "gemini", ["text", "vision", "transcription"], "Cost-efficient lightweight model"),

    # Gemini 2.5 series (stable)
    ("gemini-2.5-pro", "gemini", ["text", "vision", "transcription"], "Advanced reasoning and coding"),
    ("gemini-2.5-flash", "gemini", ["text", "vision", "transcription"], "Fast multimodal model"),
    ("gemini-2.5-flash-lite", "gemini", ["text", "vision", "transcription"], "Budget multimodal model"),

    # Gemini — Image generation
    ("imagen-4", "gemini", ["image_gen"], "Imagen 4 — text-to-image up to 2K"),
    ("gemini-3.1-flash-image-preview", "gemini", ["text", "vision", "transcription", "image_gen"], "Gemini 3.1 native image gen"),
    ("gemini-3-pro-image-preview", "gemini", ["text", "vision", "transcription", "image_gen"], "Gemini 3 Pro native image gen"),
    ("gemini-2.5-flash-image", "gemini", ["text", "vision", "transcription", "image_gen"], "Gemini 2.5 Flash image gen"),

    # Gemini — TTS
    ("gemini-2.5-flash-preview-tts", "gemini", ["tts"], "Gemini TTS — text-to-speech"),
    ("gemini-2.5-pro-preview-tts", "gemini", ["tts"], "Gemini Pro TTS — text-to-speech"),
    ("gemini-2.5-flash-lite-preview-tts", "gemini", ["tts"], "Gemini Lite TTS — budget text-to-speech"),

    # Gemini — Audio
    ("gemini-2.5-flash-native-audio-preview-12-2025", "gemini", ["text"], "Native audio — realtime voice"),

    # Deprecated Gemini models (kept for reference)
    ("gemini-2.0-flash", "gemini", ["text", "vision", "transcription"], "Gemini 2.0 Flash (deprecated)"),

    # ━━ Bedrock ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Anthropic Claude
    ("anthropic.claude-opus-4-6-v1", "bedrock", ["text", "vision"], "Claude Opus 4.6 — most capable"),
    ("anthropic.claude-sonnet-4-6", "bedrock", ["text", "vision"], "Claude Sonnet 4.6"),
    ("anthropic.claude-opus-4-5-20251101-v1:0", "bedrock", ["text", "vision"], "Claude Opus 4.5"),
    ("anthropic.claude-sonnet-4-5-20250929-v1:0", "bedrock", ["text", "vision"], "Claude Sonnet 4.5"),
    ("anthropic.claude-opus-4-1-20250805-v1:0", "bedrock", ["text", "vision"], "Claude Opus 4.1"),
    ("anthropic.claude-sonnet-4-20250514-v1:0", "bedrock", ["text", "vision"], "Claude Sonnet 4"),
    ("anthropic.claude-haiku-4-5-20251001-v1:0", "bedrock", ["text", "vision"], "Claude Haiku 4.5 — fast"),
    ("anthropic.claude-3-5-haiku-20241022-v1:0", "bedrock", ["text"], "Claude 3.5 Haiku"),

    # Amazon Nova
    ("amazon.nova-premier-v1:0", "bedrock", ["text", "vision"], "Amazon Nova Premier — most capable"),
    ("amazon.nova-pro-v1:0", "bedrock", ["text", "vision"], "Amazon Nova Pro"),
    ("amazon.nova-2-lite-v1:0", "bedrock", ["text", "vision"], "Amazon Nova 2 Lite"),
    ("amazon.nova-lite-v1:0", "bedrock", ["text", "vision"], "Amazon Nova Lite"),
    ("amazon.nova-micro-v1:0", "bedrock", ["text"], "Amazon Nova Micro — text only"),
    ("amazon.nova-canvas-v1:0", "bedrock", ["image_gen"], "Amazon Nova Canvas — image gen"),
    ("amazon.nova-sonic-v1:0", "bedrock", ["text"], "Amazon Nova Sonic — speech I/O"),
    ("amazon.nova-2-sonic-v1:0", "bedrock", ["text"], "Amazon Nova 2 Sonic — speech I/O"),

    # Meta Llama
    ("meta.llama4-maverick-17b-instruct-v1:0", "bedrock", ["text", "vision"], "Llama 4 Maverick 17B"),
    ("meta.llama4-scout-17b-instruct-v1:0", "bedrock", ["text", "vision"], "Llama 4 Scout 17B"),
    ("meta.llama3-3-70b-instruct-v1:0", "bedrock", ["text"], "Llama 3.3 70B"),
    ("meta.llama3-2-90b-instruct-v1:0", "bedrock", ["text", "vision"], "Llama 3.2 90B"),
    ("meta.llama3-2-11b-instruct-v1:0", "bedrock", ["text", "vision"], "Llama 3.2 11B"),

    # Mistral
    ("mistral.mistral-large-3-675b-instruct", "bedrock", ["text", "vision"], "Mistral Large 3 675B"),
    ("mistral.devstral-2-123b", "bedrock", ["text"], "Devstral 2 — coding model"),
    ("mistral.magistral-small-2509", "bedrock", ["text", "vision"], "Magistral Small"),
    ("mistral.pixtral-large-2502-v1:0", "bedrock", ["text", "vision"], "Pixtral Large — vision"),

    # Mistral — additional
    ("mistral.ministral-3-14b-instruct", "bedrock", ["text", "vision"], "Ministral 14B 3.0"),
    ("mistral.ministral-3-8b-instruct", "bedrock", ["text", "vision"], "Ministral 3 8B"),

    # Cohere
    ("cohere.command-r-plus-v1:0", "bedrock", ["text"], "Cohere Command R+"),
    ("cohere.command-r-v1:0", "bedrock", ["text"], "Cohere Command R"),

    # DeepSeek
    ("deepseek.r1-v1:0", "bedrock", ["text"], "DeepSeek R1 — reasoning"),
    ("deepseek.v3-v1:0", "bedrock", ["text"], "DeepSeek V3.1"),
    ("deepseek.v3.2", "bedrock", ["text"], "DeepSeek V3.2"),

    # AI21
    ("ai21.jamba-1-5-large-v1:0", "bedrock", ["text"], "AI21 Jamba 1.5 Large"),
    ("ai21.jamba-1-5-mini-v1:0", "bedrock", ["text"], "AI21 Jamba 1.5 Mini"),
]

# ── Thinking capability metadata ──────────────────────────────────────────────
# Maps model name → thinking type:
#   "effort"  → OpenAI reasoning_effort (low/medium/high)
#   "budget"  → Gemini thinking_budget (tokens) or Claude budget_tokens

THINKING_MODELS: dict[str, str] = {
    # OpenAI o-series (reasoning_effort)
    "o1": "effort",
    "o3": "effort",
    "o3-pro": "effort",
    "o3-mini": "effort",
    "o4-mini": "effort",
    "o3-deep-research": "effort",
    "o4-mini-deep-research": "effort",

    # Gemini 2.5 + 3.x (thinking_budget)
    "gemini-2.5-pro": "budget",
    "gemini-2.5-flash": "budget",
    "gemini-2.5-flash-lite": "budget",
    "gemini-3.1-pro-preview": "budget",
    "gemini-3-flash-preview": "budget",
    "gemini-3.1-flash-lite-preview": "budget",
    "gemini-3.1-flash-image-preview": "budget",
    "gemini-3-pro-image-preview": "budget",

    # Claude on Bedrock (budget_tokens)
    "anthropic.claude-opus-4-6-v1": "budget",
    "anthropic.claude-sonnet-4-6": "budget",
    "anthropic.claude-opus-4-5-20251101-v1:0": "budget",
    "anthropic.claude-sonnet-4-5-20250929-v1:0": "budget",
    "anthropic.claude-opus-4-1-20250805-v1:0": "budget",
    "anthropic.claude-sonnet-4-20250514-v1:0": "budget",
    "anthropic.claude-haiku-4-5-20251001-v1:0": "budget",
}

# Build exact-match lookup from catalog
EXACT_MODELS: dict[str, str] = {name: provider for name, provider, _, _ in MODEL_CATALOG}

# Prefix rules for models not in the catalog (checked in order)
PREFIX_RULES: list[tuple[str, str]] = [
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("chatgpt-", "openai"),
    ("dall-e", "openai"),
    ("tts-", "openai"),
    ("whisper", "openai"),
    ("gemini-", "gemini"),
    ("imagen-", "gemini"),
    ("anthropic.", "bedrock"),
    ("amazon.", "bedrock"),
    ("meta.", "bedrock"),
    ("mistral.", "bedrock"),
    ("cohere.", "bedrock"),
    ("ai21.", "bedrock"),
    ("deepseek.", "bedrock"),
]

_provider_cache: dict[str, Provider] = {}


def _create_provider(name: str) -> Provider:
    if name in _provider_cache:
        return _provider_cache[name]

    if name == "openai":
        from ocean.providers.openai import OpenAIProvider
        provider = OpenAIProvider()
    elif name == "gemini":
        from ocean.providers.gemini import GeminiProvider
        provider = GeminiProvider()
    elif name == "bedrock":
        from ocean.providers.bedrock import BedrockProvider
        provider = BedrockProvider()
    else:
        raise ModelNotFoundError(f"Unknown provider: {name}")

    _provider_cache[name] = provider
    return provider


def resolve(model: str) -> Provider:
    model_lower = model.lower()

    # Exact match
    if model_lower in EXACT_MODELS:
        return _create_provider(EXACT_MODELS[model_lower])

    # Prefix match
    for prefix, provider_name in PREFIX_RULES:
        if model_lower.startswith(prefix):
            return _create_provider(provider_name)

    raise ModelNotFoundError(
        f"Cannot find a provider for model '{model}'. "
        "Use a model name like gpt-5.4, gemini-2.5-flash, or anthropic.claude-*"
    )


def get_thinking_type(model: str) -> str | None:
    """Return 'effort' (OpenAI) or 'budget' (Gemini/Claude), or None."""
    return THINKING_MODELS.get(model.lower())


def list_models(
    provider: str | None = None,
    modality: str | None = None,
) -> list[dict]:
    """Return catalog entries, optionally filtered by provider or modality."""
    results = []
    for name, prov, modalities, description in MODEL_CATALOG:
        if provider and prov != provider.lower():
            continue
        if modality and modality not in modalities:
            continue
        results.append({
            "model": name,
            "provider": prov,
            "modalities": modalities,
            "description": description,
            "thinking": THINKING_MODELS.get(name),
        })
    return results


def get_providers() -> list[str]:
    """Return the list of all provider names."""
    return sorted({prov for _, prov, _, _ in MODEL_CATALOG})
