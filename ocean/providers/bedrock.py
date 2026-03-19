from __future__ import annotations

from typing import Iterator

from ocean.exceptions import MissingDependencyError, ProviderError, UnsupportedModalityError
from ocean.models import Modality, OceanRequest, OceanResponse, ThinkingConfig
from ocean.providers.base import Provider, THINK_PREFIX
from ocean.utils import guess_mime, read_bytes

EFFORT_TO_BUDGET = {"low": 1024, "medium": 8000, "high": 16000}
DEFAULT_THINKING_BUDGET = 5000

# Claude models >= 3.5 require cross-region inference profiles on Bedrock.
# Map AWS region prefixes to the profile geo-prefix.
_REGION_TO_PROFILE_PREFIX = {
    "us-": "us",
    "eu-": "eu",
    "ap-": "ap",
}
# Models that need a cross-region inference profile (Claude 3.5+ / 4.x)
_NEEDS_PROFILE_PREFIXES = (
    "anthropic.claude-3-5",
    "anthropic.claude-3-7",
    "anthropic.claude-haiku-4",
    "anthropic.claude-sonnet-4",
    "anthropic.claude-opus-4",
)


def _resolve_model_id(model_id: str) -> str:
    """Prepend cross-region inference profile prefix when required."""
    # Already has a geo prefix (us. / eu. / ap.)
    if model_id[:3] in ("us.", "eu.", "ap."):
        return model_id
    # Only apply to models that need a profile
    if not any(model_id.startswith(p) for p in _NEEDS_PROFILE_PREFIXES):
        return model_id
    try:
        import boto3
        region = boto3.session.Session().region_name or ""
    except Exception:
        region = ""
    for region_prefix, geo in _REGION_TO_PROFILE_PREFIX.items():
        if region.startswith(region_prefix):
            return f"{geo}.{model_id}"
    # Default to US profile
    return f"us.{model_id}"


def _get_client():
    try:
        import boto3
    except ImportError:
        raise MissingDependencyError("Bedrock", "boto3")
    return boto3.client("bedrock-runtime")


def _thinking_fields(thinking: ThinkingConfig | None) -> dict | None:
    """Return additionalModelRequestFields dict for Claude extended thinking, or None."""
    if thinking is None or thinking.enabled is False:
        return None

    budget = thinking.budget
    if budget is None and thinking.effort:
        budget = EFFORT_TO_BUDGET.get(thinking.effort, DEFAULT_THINKING_BUDGET)
    if budget is None:
        budget = DEFAULT_THINKING_BUDGET

    return {"thinking": {"type": "enabled", "budget_tokens": budget}}


def _extract_text_and_thinking(content: list[dict], show: bool) -> tuple[str, str | None]:
    """Parse Claude response content blocks into (answer, thinking_text)."""
    answer_parts: list[str] = []
    thinking_parts: list[str] = []

    for block in content:
        if block.get("type") == "thinking":
            thinking_parts.append(block.get("thinking", ""))
        elif block.get("type") == "text" or "text" in block:
            answer_parts.append(block.get("text", ""))

    thinking_text = "\n".join(thinking_parts) if thinking_parts else None
    return "".join(answer_parts), thinking_text


class BedrockProvider(Provider):
    def complete(self, request: OceanRequest) -> OceanResponse:
        if request.modality in (Modality.IMAGE_GEN, Modality.TTS):
            raise UnsupportedModalityError(f"Bedrock does not support {request.modality.value}")

        client = _get_client()
        messages = self._build_messages(request)
        kwargs: dict = {"modelId": _resolve_model_id(request.model), "messages": messages}

        inference_config: dict = {}
        if request.max_tokens is not None:
            inference_config["maxTokens"] = request.max_tokens
        if request.temperature is not None:
            inference_config["temperature"] = request.temperature
        if inference_config:
            kwargs["inferenceConfig"] = inference_config

        extra = _thinking_fields(request.thinking)
        if extra:
            kwargs["additionalModelRequestFields"] = extra

        try:
            response = client.converse(**kwargs)
        except Exception as e:
            raise ProviderError(str(e)) from e

        content = response.get("output", {}).get("message", {}).get("content", [])
        text, thinking_text = _extract_text_and_thinking(content, show=bool(request.thinking and request.thinking.show))

        usage_raw = response.get("usage", {})
        usage = None
        if usage_raw:
            usage = {
                "prompt_tokens": usage_raw.get("inputTokens"),
                "completion_tokens": usage_raw.get("outputTokens"),
            }

        return OceanResponse(text=text, thinking_text=thinking_text, model=request.model, usage=usage)

    def stream(self, request: OceanRequest) -> Iterator[str]:
        if request.modality not in (Modality.TEXT, Modality.VISION):
            raise UnsupportedModalityError(f"Bedrock does not support streaming for {request.modality.value}")

        client = _get_client()
        messages = self._build_messages(request)
        kwargs: dict = {"modelId": _resolve_model_id(request.model), "messages": messages}

        inference_config: dict = {}
        if request.max_tokens is not None:
            inference_config["maxTokens"] = request.max_tokens
        if request.temperature is not None:
            inference_config["temperature"] = request.temperature
        if inference_config:
            kwargs["inferenceConfig"] = inference_config

        extra = _thinking_fields(request.thinking)
        if extra:
            kwargs["additionalModelRequestFields"] = extra

        show = request.thinking and request.thinking.show

        try:
            response = client.converse_stream(**kwargs)
        except Exception as e:
            raise ProviderError(str(e)) from e

        in_thinking_block = False
        for event in response.get("stream", []):
            if "contentBlockStart" in event:
                block_start = event["contentBlockStart"].get("start", {})
                in_thinking_block = "reasoningContent" in block_start
            elif "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "reasoningContent" in delta:
                    text = delta["reasoningContent"].get("text", "")
                    if text and show:
                        yield THINK_PREFIX + text
                elif "text" in delta:
                    text = delta.get("text", "")
                    if text:
                        yield text
            elif "contentBlockStop" in event:
                in_thinking_block = False

    def chat(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        thinking: ThinkingConfig | None = None,
    ) -> str:
        client = _get_client()
        bedrock_msgs = [
            {"role": m["role"], "content": [{"text": m["content"]}]}
            for m in messages
        ]
        kwargs: dict = {"modelId": _resolve_model_id(model), "messages": bedrock_msgs}
        if system:
            kwargs["system"] = [{"text": system}]
        inference_config: dict = {}
        if max_tokens is not None:
            inference_config["maxTokens"] = max_tokens
        if temperature is not None:
            inference_config["temperature"] = temperature
        if inference_config:
            kwargs["inferenceConfig"] = inference_config
        extra = _thinking_fields(thinking)
        if extra:
            kwargs["additionalModelRequestFields"] = extra
        try:
            response = client.converse(**kwargs)
        except Exception as e:
            raise ProviderError(str(e)) from e
        content = response.get("output", {}).get("message", {}).get("content", [])
        text, _ = _extract_text_and_thinking(content, show=False)
        return text

    def chat_stream(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        thinking: ThinkingConfig | None = None,
    ) -> Iterator[str]:
        client = _get_client()
        bedrock_msgs = [
            {"role": m["role"], "content": [{"text": m["content"]}]}
            for m in messages
        ]
        kwargs: dict = {"modelId": _resolve_model_id(model), "messages": bedrock_msgs}
        if system:
            kwargs["system"] = [{"text": system}]
        inference_config: dict = {}
        if max_tokens is not None:
            inference_config["maxTokens"] = max_tokens
        if temperature is not None:
            inference_config["temperature"] = temperature
        if inference_config:
            kwargs["inferenceConfig"] = inference_config
        extra = _thinking_fields(thinking)
        if extra:
            kwargs["additionalModelRequestFields"] = extra
        show = thinking and thinking.show
        try:
            response = client.converse_stream(**kwargs)
        except Exception as e:
            raise ProviderError(str(e)) from e

        for event in response.get("stream", []):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                if "reasoningContent" in delta:
                    text = delta["reasoningContent"].get("text", "")
                    if text and show:
                        yield THINK_PREFIX + text
                elif "text" in delta:
                    text = delta.get("text", "")
                    if text:
                        yield text

    # ── helpers ──

    def _build_messages(self, request: OceanRequest) -> list[dict]:
        content: list[dict] = []

        if request.modality == Modality.VISION and request.image:
            img_bytes = read_bytes(request.image)
            mime = guess_mime(request.image)
            fmt = mime.split("/")[-1] if "/" in mime else "png"
            content.append({"image": {"format": fmt, "source": {"bytes": img_bytes}}})

        if request.modality == Modality.TRANSCRIPTION and request.audio:
            raise UnsupportedModalityError("Bedrock does not support audio transcription")

        if request.prompt:
            content.append({"text": request.prompt})

        return [{"role": "user", "content": content}]
