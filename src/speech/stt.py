"""Speech-to-text via Gemini's native audio understanding.

There's no dedicated "STT endpoint" in Gemini's API the way Whisper is one
-- audio is just another input modality to the same model. We ask it to
transcribe verbatim and treat the reply as the transcript. Kept as its own
call (rather than combining transcription + a reply into one request) so
we always have an explicit text transcript to log, print, and later feed
into scene memory -- not just an implicit intermediate the model used on
its way to a response.

Measured live against the real API (free-tier gemini-3.8-flash, a ~3.6s
clip): 33.6s. Audio-modality calls on the free tier are genuinely slow,
not a bug -- the 20s timeout this module started with was silently
swallowing the response before it came back. See CharacterOrchestrator's
"thinking" indicator, which exists because of this measurement.
"""

from __future__ import annotations

import base64

from .gemini_client import TEXT_MODEL, get_client

_TRANSCRIBE_PROMPT = (
    "Transcribe this short audio clip verbatim, in the language it was "
    "spoken. Reply with ONLY the transcript text -- no quotes, no "
    "preamble, no commentary. If there is no speech at all, reply with "
    "an empty string."
)


def transcribe(wav_bytes: bytes, timeout_s: float = 60.0) -> str:
    client = get_client()
    interaction = client.interactions.create(
        model=TEXT_MODEL,
        input=[
            {"type": "text", "text": _TRANSCRIBE_PROMPT},
            {
                "type": "audio",
                "data": base64.b64encode(wav_bytes).decode("utf-8"),
                "mime_type": "audio/wav",
            },
        ],
        timeout=timeout_s,
    )
    return (interaction.output_text or "").strip()
