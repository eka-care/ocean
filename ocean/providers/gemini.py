from __future__ import annotations

import wave
from typing import Iterator

from ocean.exceptions import MissingDependencyError, ProviderError
from ocean.models import Modality, OceanRequest, OceanResponse, ThinkingConfig
from ocean.providers.base import Provider, THINK_PREFIX
from ocean.utils import guess_mime, read_bytes, write_file_bytes

IMAGEN_MODELS = {"imagen-4", "imagen-3.0-generate-002", "imagen-4.0-generate-001", "imagen-4.0-fast-generate-001"}

# Effort → thinking budget token mapping
EFFORT_TO_BUDGET = {"low": 1024, "medium": 8000, "high": 16000}


def _get_client():
    import os

    try:
        from google import genai
    except ImportError:
        raise MissingDependencyError("Gemini", "google-genai")
    api_key = os.environ.get("GOOGLE_AI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key) if api_key else genai.Client()


def _build_thinking_config(thinking: ThinkingConfig | None):
    """Return a google.genai.types.ThinkingConfig or None."""
    if thinking is None:
        return None

    from google.genai import types

    if thinking.enabled is False:
        return types.ThinkingConfig(thinking_budget=0)

    budget = thinking.budget
    if budget is None and thinking.effort:
        budget = EFFORT_TO_BUDGET.get(thinking.effort)

    kwargs = {}
    if budget is not None:
        kwargs["thinking_budget"] = budget
    if thinking.show:
        kwargs["include_thoughts"] = True

    return types.ThinkingConfig(**kwargs) if kwargs else None


class GeminiProvider(Provider):
    def complete(self, request: OceanRequest) -> OceanResponse:
        client = _get_client()

        if request.modality == Modality.IMAGE_GEN:
            return self._image_gen(client, request)
        if request.modality == Modality.TTS:
            return self._tts(client, request)

        contents = self._build_contents(request)
        config = self._build_config(request)

        try:
            response = client.models.generate_content(
                model=request.model, contents=contents, config=config,
            )
        except Exception as e:
            raise ProviderError(str(e)) from e

        # Separate thinking text from answer text
        thinking_text: str | None = None
        answer_parts: list[str] = []
        try:
            for part in response.candidates[0].content.parts:
                if getattr(part, "thought", False):
                    thinking_text = (thinking_text or "") + (part.text or "")
                elif part.text:
                    answer_parts.append(part.text)
        except Exception:
            answer_parts = [response.text or ""]

        return OceanResponse(
            text="".join(answer_parts) or response.text,
            thinking_text=thinking_text,
            model=request.model,
        )

    def stream(self, request: OceanRequest) -> Iterator[str]:
        if request.modality not in (Modality.TEXT, Modality.VISION, Modality.TRANSCRIPTION):
            resp = self.complete(request)
            if resp.text:
                yield resp.text
            return

        client = _get_client()
        contents = self._build_contents(request)
        config = self._build_config(request)

        show = request.thinking and request.thinking.show

        try:
            for chunk in client.models.generate_content_stream(
                model=request.model, contents=contents, config=config,
            ):
                try:
                    for part in chunk.candidates[0].content.parts:
                        if getattr(part, "thought", False):
                            if show and part.text:
                                yield THINK_PREFIX + part.text
                        elif part.text:
                            yield part.text
                except Exception:
                    if chunk.text:
                        yield chunk.text
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
        contents = self._messages_to_contents(messages)
        config = self._chat_config(max_tokens, temperature, system, thinking)
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config,
            )
        except Exception as e:
            raise ProviderError(str(e)) from e
        return response.text or ""

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
        contents = self._messages_to_contents(messages)
        config = self._chat_config(max_tokens, temperature, system, thinking)
        show = thinking and thinking.show

        try:
            for chunk in client.models.generate_content_stream(
                model=model, contents=contents, config=config,
            ):
                try:
                    for part in chunk.candidates[0].content.parts:
                        if getattr(part, "thought", False):
                            if show and part.text:
                                yield THINK_PREFIX + part.text
                        elif part.text:
                            yield part.text
                except Exception:
                    if chunk.text:
                        yield chunk.text
        except Exception as e:
            raise ProviderError(str(e)) from e

    # ── helpers ──

    @staticmethod
    def _messages_to_contents(messages: list[dict]) -> list[dict]:
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        return contents

    @staticmethod
    def _chat_config(max_tokens, temperature, system, thinking: ThinkingConfig | None = None):
        from google.genai import types

        kwargs = {}
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if system:
            kwargs["system_instruction"] = system

        thinking_cfg = _build_thinking_config(thinking)
        if thinking_cfg is not None:
            kwargs["thinking_config"] = thinking_cfg

        return types.GenerateContentConfig(**kwargs) if kwargs else None

    def _build_contents(self, request: OceanRequest) -> list:
        from google.genai import types

        parts: list = []

        if request.modality == Modality.VISION and request.image:
            img_bytes = read_bytes(request.image)
            mime = guess_mime(request.image)
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))

        if request.modality == Modality.TRANSCRIPTION and request.audio:
            audio_bytes = read_bytes(request.audio)
            mime = guess_mime(request.audio)
            parts.append(types.Part.from_bytes(data=audio_bytes, mime_type=mime))

        if request.prompt:
            parts.append(request.prompt)

        return parts

    def _build_config(self, request: OceanRequest):
        from google.genai import types

        kwargs = {}
        if request.max_tokens is not None:
            kwargs["max_output_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature

        thinking_cfg = _build_thinking_config(request.thinking)
        if thinking_cfg is not None:
            kwargs["thinking_config"] = thinking_cfg

        return types.GenerateContentConfig(**kwargs) if kwargs else None

    def _image_gen(self, client, request: OceanRequest) -> OceanResponse:
        from google.genai import types

        model_lower = request.model.lower()

        if model_lower in IMAGEN_MODELS:
            try:
                response = client.models.generate_images(
                    model=request.model,
                    prompt=request.prompt,
                    config=types.GenerateImagesConfig(number_of_images=1),
                )
            except Exception as e:
                raise ProviderError(str(e)) from e

            if not response.generated_images:
                raise ProviderError("No images generated")

            img_bytes = response.generated_images[0].image.image_bytes
            out = request.output or "output.png"
            path = write_file_bytes(out, img_bytes)
            return OceanResponse(text=f"Image saved to {path}", file_path=path, model=request.model)

        try:
            response = client.models.generate_content(
                model=request.model,
                contents=request.prompt,
                config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
            )
        except Exception as e:
            raise ProviderError(str(e)) from e

        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                img_bytes = part.inline_data.data
                out = request.output or "output.png"
                path = write_file_bytes(out, img_bytes)
                return OceanResponse(text=f"Image saved to {path}", file_path=path, model=request.model)

        raise ProviderError("No image data in response")

    def _tts(self, client, request: OceanRequest) -> OceanResponse:
        from google.genai import types

        voice = request.voice or "Kore"
        try:
            response = client.models.generate_content(
                model=request.model,
                contents=request.prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                        )
                    ),
                ),
            )
        except Exception as e:
            raise ProviderError(str(e)) from e

        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("audio/"):
                pcm_data = part.inline_data.data
                out = request.output or "output.wav"
                if part.inline_data.mime_type == "audio/L16" or out.endswith(".wav"):
                    wav_bytes = self._pcm_to_wav(pcm_data, sample_rate=24000)
                    path = write_file_bytes(out, wav_bytes)
                else:
                    path = write_file_bytes(out, pcm_data)
                return OceanResponse(text=f"Audio saved to {path}", file_path=path, model=request.model)

        raise ProviderError("No audio data in response")

    @staticmethod
    def _pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000) -> bytes:
        import io

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return buf.getvalue()
