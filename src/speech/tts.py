"""Text-to-speech via Gemini's native audio generation.

Returns proper WAV bytes -- but the model doesn't hand those back
directly. Checked live: gemini-2.5-flash-preview-tts's response is
headerless raw PCM (mime_type "audio/L16;codec=pcm;rate=24000", i.e. no
RIFF header at all), which is why playback failed with "Format not
recognised" the first time this was wired up against a different model
(gemini-3.8-flash-tts, whose docs describe a WAV-wrapped response) before
switching models for the free-tier quota reasons in gemini_client.py. The
response does carry its own sample_rate/channels, so we trust those
rather than hardcoding 24kHz mono, in case that changes if the model is
ever swapped again.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import soundfile as sf

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
    audio = interaction.output_audio
    pcm_bytes = base64.b64decode(audio.data)
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    channels = getattr(audio, "channels", None) or 1
    sample_rate = getattr(audio, "sample_rate", None) or 24000
    if channels > 1:
        samples = samples.reshape(-1, channels)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
