"""OceanClient — reusable client for interacting with AI models."""

from __future__ import annotations

import io
from typing import Iterator, Union

from ocean.models import OceanRequest, OceanResponse
from ocean.registry import list_models, resolve
from ocean.utils import resolve_modality

BinarySource = Union[str, bytes, io.BytesIO]


class OceanClient:
    """A stateful client that holds default options for AI requests.

    Usage:
        client = OceanClient(max_tokens=100, temperature=0.7)
        response = client.invoke("gpt-4o", "What is the capital of France?")
        print(response.text)

        # Stream
        for chunk in client.invoke("gemini-2.0-flash", "Tell me a joke", stream=True):
            print(chunk, end="")

        # Vision
        response = client.invoke("gpt-4o", "Describe this image", image="photo.png")

        # Image generation
        response = client.invoke("dall-e-3", "a sunset", output="sunset.png")

        # TTS
        response = client.invoke("tts-1", "Hello world", output="hello.mp3", voice="nova")

        # Transcription
        response = client.invoke("whisper-1", audio="recording.mp3")

        # List available models
        models = client.models()
        models = client.models(provider="openai")
        models = client.models(modality="tts")
    """

    def __init__(
        self,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        default_voice: str | None = None,
    ):
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._default_voice = default_voice

    def invoke(
        self,
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
    ) -> OceanResponse | Iterator[str]:
        """Send a request to any supported AI model.

        Per-call kwargs override client defaults.
        """
        modality = resolve_modality(model, image=image, audio=audio)

        request = OceanRequest(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
            temperature=temperature if temperature is not None else self._temperature,
            stream=stream,
            modality=modality,
            image=image,
            audio=audio,
            output=output,
            voice=voice or self._default_voice,
        )

        provider = resolve(model)

        if stream:
            return provider.stream(request)
        return provider.complete(request)

    def chat(self, model: str, system: str | None = None) -> "ChatSession":
        """Start an interactive chat session.

        Usage:
            session = client.chat("gpt-5.4")
            reply = session.send("Hello!")
            reply = session.send("Tell me more")

            # Streaming
            for chunk in session.send("Explain quantum computing", stream=True):
                print(chunk, end="")
        """
        return ChatSession(
            model=model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system,
        )

    @staticmethod
    def models(
        provider: str | None = None,
        modality: str | None = None,
    ) -> list[dict]:
        """List supported models, optionally filtered by provider or modality."""
        return list_models(provider=provider, modality=modality)


class ChatSession:
    """A multi-turn conversation session with a model."""

    def __init__(
        self,
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system = system
        self.messages: list[dict] = []
        self._provider = resolve(model)

    def send(self, message: str, *, stream: bool = False) -> str | Iterator[str]:
        """Send a message and get the reply.

        Returns a string, or an iterator of strings if stream=True.
        """
        self.messages.append({"role": "user", "content": message})

        if stream:
            return self._stream_reply()

        reply = self._provider.chat(
            messages=self.messages,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system,
        )
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def _stream_reply(self) -> Iterator[str]:
        chunks: list[str] = []
        for chunk in self._provider.chat_stream(
            messages=self.messages,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system,
        ):
            chunks.append(chunk)
            yield chunk
        self.messages.append({"role": "assistant", "content": "".join(chunks)})

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()

    @property
    def history(self) -> list[dict]:
        """Return a copy of the conversation history."""
        return list(self.messages)
