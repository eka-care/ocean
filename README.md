# Ocean

Unified AI CLI and Python library. One interface for OpenAI, Google Gemini, and AWS Bedrock.

## Install

```bash
# All providers
uv pip install -e ".[all]"

# Or selectively
uv pip install -e ".[openai]"
uv pip install -e ".[gemini]"
uv pip install -e ".[bedrock]"
```

## Credentials

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=sk-...
GOOGLE_AI_API_KEY=...
AWS_DEFAULT_REGION=us-east-1
```

For Bedrock, standard AWS credential sources work (env vars, `~/.aws/credentials`, IAM roles).

## CLI

```bash
# Text
ocean gpt-4o "explain transformers"
ocean gemini-2.5-flash "summarize this" --max-tokens 200

# Vision
ocean gpt-4o "describe this image" --image photo.png

# Image generation
ocean dall-e-3 "a sunset over mountains" -o sunset.png
ocean imagen-4.0-generate-001 "a cat" -o cat.png

# Audio transcription
ocean whisper-1 --audio recording.mp3
ocean gemini-2.5-flash "transcribe" --audio meeting.mp3

# Text-to-speech
ocean tts-1 "Hello world" -o hello.mp3 --voice nova
ocean gemini-2.5-flash-preview-tts "Hello" -o hello.wav --voice Kore

# Reasoning / thinking
ocean o3 "solve this step by step" --thinking-effort high
ocean gemini-2.5-flash "hard math problem" --thinking-budget 8000 --show-thinking
ocean anthropic.claude-sonnet-4-5-20250929-v1:0 "complex task" --thinking

# Interactive chat
ocean chat gpt-4o
ocean chat gemini-2.5-flash --system "You are a helpful tutor"

# List models
ocean list
ocean list --provider openai
ocean list --modality tts
ocean list --json
```

### Chat commands

Inside `ocean chat`, use slash commands:

| Command | Description |
|---------|-------------|
| `/clear` | Clear conversation history |
| `/system <prompt>` | Set system prompt |
| `/history` | Show conversation history |
| `/model` | Show current model |
| `/exit` | Exit |

## Python Library

```python
from ocean import invoke, OceanClient

# One-shot
response = invoke("gpt-4o", "What is the capital of France?")
print(response.text)

# Stream
for chunk in invoke("gemini-2.5-flash", "Tell me a story", stream=True):
    print(chunk, end="", flush=True)

# Vision
response = invoke("gpt-4o", "Describe this image", image="photo.png")

# Pass bytes or BytesIO directly
audio_bytes = open("recording.mp3", "rb").read()
response = invoke("whisper-1", audio=audio_bytes)

import io
img_bytes = open("photo.png", "rb").read()
response = invoke("gpt-4o", "what's in this?", image=io.BytesIO(img_bytes))

# Image generation
response = invoke("dall-e-3", "a sunset", output="sunset.png")

# TTS
response = invoke("tts-1", "Hello world", output="hello.mp3", voice="nova")

# Thinking
from ocean import ThinkingConfig
response = invoke(
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "solve this",
    thinking=ThinkingConfig(enabled=True, budget=8000, show=True),
)
print(response.thinking_text)
print(response.text)
```

### OceanClient

Holds default settings across multiple calls:

```python
from ocean import OceanClient

client = OceanClient(max_tokens=500, temperature=0.7)

response = client.ask("gpt-4o", "hello")
models = client.models(provider="openai", modality="text")
```

### ChatSession

Multi-turn conversations:

```python
session = client.chat("gpt-4o", system="You are a pirate")

reply = session.send("Hello!")
print(reply)

# Stream
for chunk in session.send("Tell me a story", stream=True):
    print(chunk, end="")

session.clear()           # reset history
print(session.history)    # list of {role, content} dicts
```

## Supported Providers

| Provider | Text | Vision | Image Gen | Transcription | TTS | Thinking |
|----------|------|--------|-----------|---------------|-----|----------|
| OpenAI | ✓ | ✓ | ✓ (DALL-E) | ✓ (Whisper) | ✓ | ✓ (o-series) |
| Gemini | ✓ | ✓ | ✓ (Imagen) | ✓ | ✓ | ✓ (2.5+) |
| Bedrock | ✓ | ✓ | — | — | — | ✓ (Claude) |

Run `ocean list` for the full model catalog.
