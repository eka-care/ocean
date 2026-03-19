import io
from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict

# A binary source can be a file path, raw bytes, or a BytesIO object
BinarySource = Union[str, bytes, io.BytesIO]


class Modality(str, Enum):
    TEXT = "text"
    VISION = "vision"
    IMAGE_GEN = "image_gen"
    TRANSCRIPTION = "transcription"
    TTS = "tts"


class ThinkingConfig(BaseModel):
    """Unified thinking / reasoning configuration across providers.

    OpenAI o-series   → reasoning_effort (low/medium/high)
    Gemini 2.5+/3.x   → thinking_budget tokens; budget=0 disables; -1 = dynamic
    Claude on Bedrock → additionalModelRequestFields budget_tokens
    """
    enabled: Optional[bool] = None          # None = provider default
    effort: Optional[str] = None            # "low" | "medium" | "high"
    budget: Optional[int] = None            # explicit token budget
    show: bool = False                      # surface thinking text in output


class OceanRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str
    prompt: str = ""
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    modality: Modality = Modality.TEXT
    thinking: Optional[ThinkingConfig] = None

    # Multimodal inputs: file path, raw bytes, or BytesIO
    image: Optional[BinarySource] = None
    audio: Optional[BinarySource] = None
    output: Optional[str] = None
    voice: Optional[str] = None


class OceanResponse(BaseModel):
    text: Optional[str] = None
    thinking_text: Optional[str] = None    # surfaced when show=True
    file_path: Optional[str] = None
    model: str = ""
    usage: Optional[dict] = None
