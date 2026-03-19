import io
import mimetypes
from pathlib import Path
from typing import Union

from ocean.models import Modality

BinarySource = Union[str, bytes, io.BytesIO]

# Models that produce images
IMAGE_GEN_PREFIXES = ("dall-e", "imagen-", "gpt-image-", "chatgpt-image-")
IMAGE_GEN_SUFFIXES = ("-image", "-image-preview")

# Models that transcribe audio
TRANSCRIPTION_MODELS = {"whisper-1"}
TRANSCRIPTION_SUFFIXES = ("-transcribe",)

# Models that produce speech
TTS_PREFIXES = ("tts-",)
TTS_SUFFIXES = ("-tts",)


def resolve_modality(
    model: str,
    image: "BinarySource | None" = None,
    audio: "BinarySource | None" = None,
) -> Modality:
    model_lower = model.lower()

    # Explicit image generation models
    if any(model_lower.startswith(p) for p in IMAGE_GEN_PREFIXES):
        return Modality.IMAGE_GEN
    if any(model_lower.endswith(s) for s in IMAGE_GEN_SUFFIXES):
        return Modality.IMAGE_GEN

    # Transcription models
    if model_lower in TRANSCRIPTION_MODELS:
        return Modality.TRANSCRIPTION
    if any(model_lower.endswith(s) for s in TRANSCRIPTION_SUFFIXES):
        return Modality.TRANSCRIPTION

    # TTS models
    if any(model_lower.startswith(p) for p in TTS_PREFIXES):
        return Modality.TTS
    if any(model_lower.endswith(s) for s in TTS_SUFFIXES):
        return Modality.TTS

    # Vision if image flag is present
    if image:
        return Modality.VISION

    # Audio input on a non-transcription model → treat as vision-like multimodal
    if audio:
        return Modality.TRANSCRIPTION

    return Modality.TEXT


def read_file_bytes(path: str) -> bytes:
    return Path(path).expanduser().read_bytes()


def read_bytes(source: "BinarySource") -> bytes:
    """Read bytes from a file path, bytes object, or BytesIO."""
    if isinstance(source, bytes):
        return source
    if isinstance(source, io.BytesIO):
        pos = source.tell()
        data = source.read()
        source.seek(pos)  # restore position for potential re-reads
        return data
    return Path(source).expanduser().read_bytes()


def _guess_mime_from_bytes(data: bytes) -> str:
    """Detect MIME type from magic bytes for common image and audio formats."""
    if data[:4] == b"\x89PNG":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if data[:3] == b"ID3" or (len(data) >= 2 and data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")):
        return "audio/mpeg"
    if data[:4] == b"OggS":
        return "audio/ogg"
    if data[:4] == b"fLaC":
        return "audio/flac"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "audio/mp4"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "audio/webm"
    return "application/octet-stream"


def guess_mime(source: "BinarySource") -> str:
    """Guess MIME type from a file path (by extension) or binary data (magic bytes)."""
    if isinstance(source, str):
        mime, _ = mimetypes.guess_type(source)
        return mime or "application/octet-stream"
    return _guess_mime_from_bytes(read_bytes(source))


def write_file_bytes(path: str, data: bytes) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return str(p)
