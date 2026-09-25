"""Text-to-speech via Gemini's native audio generation.

Uses stream=True. Measured live against the real API: the exact same
request, streaming vs. not, was 3.8s vs ~16s -- not because chunks arrive
incrementally (a short reply came back as a single audio event either
way), but because the streaming code path itself is faster server-side.
For a short reply this stops mattering once it's noticeably shorter than
the ~20-30s audio-understanding call anyway, but it's a real, measured,
no-cost win, so there's no reason not to take it.

Returns proper WAV bytes -- but the model doesn't hand those back
directly. Checked live: gemini-2.5-flash-preview-tts's response is
headerless raw PCM (mime_type "audio/L16;codec=pcm;rate=24000", i.e. no
RIFF header at all), which is why playback failed with "Format not
recognised" the first time this was wired up against a different model
(gemini-3.8-flash-tts, whose docs describe a WAV-wrapped response) before
switching models for the free-tier quota reasons in gemini_client.py. The
response carries its own sample_rate/channels per chunk, so those are
trusted rather than hardcoding 24kHz mono.
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
    stream = client.interactions.create(
        model=TTS_MODEL,
        input=[
            {
                "type": "user_input",
                "content": [{"type": "text", "text": text}],
            }
        ],
        response_format={"type": "audio"},
        generation_config={"speech_config": [{"voice": voice}]},
        stream=True,
        timeout=timeout_s,
    )

    pcm_chunks: list[bytes] = []
    sample_rate = 24000
    channels = 1
    for event in stream:
        delta = getattr(event, "delta", None)
        if delta is None or getattr(delta, "type", None) != "audio":
            continue
        pcm_chunks.append(base64.b64decode(delta.data or ""))
        sample_rate = delta.sample_rate or sample_rate
        channels = delta.channels or channels

    samples = np.frombuffer(b"".join(pcm_chunks), dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
