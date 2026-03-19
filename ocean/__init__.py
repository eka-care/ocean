"""Ocean — Unified AI CLI & Library."""

from __future__ import annotations

import io
from typing import Iterator, Union

from dotenv import load_dotenv

load_dotenv()

from ocean.client import ChatSession, OceanClient
from ocean.models import Modality, OceanRequest, OceanResponse, ThinkingConfig
from ocean.registry import resolve
from ocean.utils import resolve_modality

BinarySource = Union[str, bytes, io.BytesIO]


def invoke(
    model: str,
    prompt: str = "",
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    stream: bool = False,
    image: "BinarySource | None" = None,
    audio: "BinarySource | None" = None,
    output: str | None = None,
    voice: str | None = None,
    thinking: ThinkingConfig | None = None,
) -> OceanResponse | Iterator[str]:
    """Send a request to any supported AI model.

    Returns an OceanResponse, or an iterator of strings if stream=True.
    """
    modality = resolve_modality(model, image=image, audio=audio)

    request = OceanRequest(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
        modality=modality,
        image=image,
        audio=audio,
        output=output,
        voice=voice,
        thinking=thinking,
    )

    provider = resolve(model)

    if stream:
        return provider.stream(request)
    return provider.complete(request)
