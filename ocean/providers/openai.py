from __future__ import annotations

import base64
from typing import Iterator

from ocean.exceptions import MissingDependencyError, ProviderError
from ocean.models import Modality, OceanRequest, OceanResponse, ThinkingConfig
from ocean.providers.base import Provider
from ocean.utils import guess_mime, read_bytes, read_file_bytes, write_file_bytes

# o-series models that support reasoning_effort
O_SERIES = {"o1", "o3", "o3-pro", "o3-mini", "o4-mini", "o3-deep-research", "o4-mini-deep-research"}


def _get_client():
    try:
        from openai import OpenAI
    except ImportError:
        raise MissingDependencyError("OpenAI", "openai")
    return OpenAI()


def _apply_thinking(kwargs: dict, model: str, thinking: ThinkingConfig | None) -> None:
    """Inject reasoning_effort for o-series models."""
    if model.lower() not in O_SERIES:
        return
    if thinking is None:
        return
    effort = thinking.effort
    if thinking.enabled is False:
        # Can't fully disable on o-series, but set lowest effort
        effort = "low"
    if effort:
        kwargs["reasoning_effort"] = effort


class OpenAIProvider(Provider):
    def complete(self, request: OceanRequest) -> OceanResponse:
        client = _get_client()

        if request.modality == Modality.IMAGE_GEN:
            return self._image_gen(client, request)
        if request.modality == Modality.TRANSCRIPTION:
            return self._transcribe(client, request)
        if request.modality == Modality.TTS:
            return self._tts(client, request)

        messages = self._build_messages(request)
        kwargs: dict = {"model": request.model, "messages": messages}
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        _apply_thinking(kwargs, request.model, request.thinking)

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            raise ProviderError(str(e)) from e

        choice = response.choices[0]
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
        return OceanResponse(text=choice.message.content, model=response.model, usage=usage)

    def stream(self, request: OceanRequest) -> Iterator[str]:
        if request.modality not in (Modality.TEXT, Modality.VISION):
            resp = self.complete(request)
            if resp.text:
                yield resp.text
            return

        client = _get_client()
        messages = self._build_messages(request)
        kwargs: dict = {"model": request.model, "messages": messages, "stream": True}
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        _apply_thinking(kwargs, request.model, request.thinking)

        try:
            for chunk in client.chat.completions.create(**kwargs):
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            raise ProviderError(str(e)) from e

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
        msgs = self._prepend_system(messages, system)
        kwargs: dict = {"model": model, "messages": msgs}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        _apply_thinking(kwargs, model, thinking)
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            raise ProviderError(str(e)) from e
        return response.choices[0].message.content or ""

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
        msgs = self._prepend_system(messages, system)
        kwargs: dict = {"model": model, "messages": msgs, "stream": True}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        _apply_thinking(kwargs, model, thinking)
        try:
            for chunk in client.chat.completions.create(**kwargs):
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            raise ProviderError(str(e)) from e

    @staticmethod
    def _prepend_system(messages: list[dict], system: str | None) -> list[dict]:
        if system:
            return [{"role": "system", "content": system}] + messages
        return messages

    # ── helpers ──

    def _build_messages(self, request: OceanRequest) -> list[dict]:
        if request.modality == Modality.VISION and request.image:
            image_data = base64.b64encode(read_bytes(request.image)).decode()
            mime = guess_mime(request.image)
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{image_data}"},
                        },
                    ],
                }
            ]
        return [{"role": "user", "content": request.prompt}]

    def _image_gen(self, client, request: OceanRequest) -> OceanResponse:
        try:
            response = client.images.generate(
                model=request.model,
                prompt=request.prompt,
                n=1,
                response_format="b64_json",
            )
        except Exception as e:
            raise ProviderError(str(e)) from e

        b64 = response.data[0].b64_json
        image_bytes = base64.b64decode(b64)
        out = request.output or "output.png"
        path = write_file_bytes(out, image_bytes)
        return OceanResponse(text=f"Image saved to {path}", file_path=path, model=request.model)

    def _transcribe(self, client, request: OceanRequest) -> OceanResponse:
        import io as _io
        audio = request.audio
        if not audio:
            raise ProviderError("No --audio file provided for transcription")
        try:
            if isinstance(audio, str):
                with open(audio, "rb") as f:
                    transcript = client.audio.transcriptions.create(model=request.model, file=f)
            else:
                data = read_bytes(audio)
                transcript = client.audio.transcriptions.create(
                    model=request.model,
                    file=("audio", _io.BytesIO(data), guess_mime(audio)),
                )
        except Exception as e:
            raise ProviderError(str(e)) from e
        return OceanResponse(text=transcript.text, model=request.model)

    def _tts(self, client, request: OceanRequest) -> OceanResponse:
        voice = request.voice or "alloy"
        try:
            response = client.audio.speech.create(
                model=request.model, voice=voice, input=request.prompt,
            )
        except Exception as e:
            raise ProviderError(str(e)) from e
        out = request.output or "output.mp3"
        path = write_file_bytes(out, response.content)
        return OceanResponse(text=f"Audio saved to {path}", file_path=path, model=request.model)
