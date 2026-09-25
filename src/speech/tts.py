"""Text-to-speech via Gemini's native audio generation.

Returns WAV bytes directly (Gemini's default for a non-streaming request:
RIFF-header WAV, 24kHz mono 16-bit PCM) -- no separate decoding step
needed before handing it to playback.play_bytes().
"""

from __future__ import annotations

import base64

from .gemini_client import TTS_MODEL, get_client

DEFAULT_VOICE = "Kore"


def synthesize(text: str, voice: str = DEFAULT_VOICE, timeout_s: float = 45.0) -> bytes:
    client = get_client()
    interaction = client.interactions.create(
        model=TTS_MODEL,
        input=[
            {
                "type": "user_input",
                "content": [{"type": "text", "text": text}],
            }
        ],
        response_format={"type": "audio"},
        generation_config={"speech_config": [{"voice": voice}]},
        timeout=timeout_s,
    )
    return base64.b64decode(interaction.output_audio.data)
