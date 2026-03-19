from abc import ABC, abstractmethod
from typing import Iterator

from ocean.models import OceanRequest, OceanResponse, ThinkingConfig

# Sentinel prefix for thinking chunks yielded by chat_stream(show=True)
THINK_PREFIX = "\x00"


class Provider(ABC):
    @abstractmethod
    def complete(self, request: OceanRequest) -> OceanResponse:
        ...

    @abstractmethod
    def stream(self, request: OceanRequest) -> Iterator[str]:
        ...

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        thinking: ThinkingConfig | None = None,
    ) -> str:
        """Send a multi-turn conversation and return the assistant reply."""
        ...

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        thinking: ThinkingConfig | None = None,
    ) -> Iterator[str]:
        """Stream a multi-turn conversation.

        When thinking.show=True, thinking chunks are prefixed with THINK_PREFIX
        so the caller can render them separately.
        """
        ...
