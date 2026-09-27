"""Text-to-speech via Gemini's native audio generation.

Conversation audio comes directly from Gemini Live; this smaller path is
only used to pre-generate a few local error phrases at startup. It uses
the API-key-compatible generate_content endpoint and returns the complete
WAV emitted by the unary Gemini 3.8 TTS response.

TTS_MODEL is gemini-3.8-flash-lite-tts, not the older gemini-2.5-flash-
preview-tts -- confirmed live the older model is capped at 10 requests/
day free tier (see gemini_client.py). An earlier version of this file
added a system_instruction to work around that older model occasionally
trying to respond conversationally to short text instead of vocalizing
it; gemini-3.8-flash-lite-tts rejects system_instruction outright
("Developer instruction is not enabled for this model") and, tested live
against the exact same previously-failing text, didn't reproduce the
original problem anyway.
"""

from __future__ import annotations

from google.genai import types

from .gemini_client import configured_tts_model, get_client

DEFAULT_VOICE = "Kore"


def synthesize(text: str, voice: str = DEFAULT_VOICE, timeout_s: float = 45.0) -> bytes:
    client = get_client()
    response = client.models.generate_content(
        model=configured_tts_model(),
        contents=[{"role": "user", "parts": [{"text": text}]}],
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config={"voice_config": {"voice": voice}},
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
        ),
    )
    for candidate in response.candidates or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and inline.data:
                # Unary generate_content returns a complete WAV container
                # for Gemini 3.8 TTS, ready for soundfile/playback.
                return inline.data
    raise RuntimeError("Gemini TTS returned no audio")
