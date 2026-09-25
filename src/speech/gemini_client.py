"""Shared Gemini client factory.

One vendor, one API key (see .env.example) for everything that isn't pure
local signal processing: vision/dialogue/tool-use, "STT" (native audio
understanding), and TTS (native audio generation). Chosen specifically
because its free tier (Flash models) needs no credit card, unlike OpenAI
(no free tier at all) or an ongoing Anthropic free tier (only a one-time
no-card trial credit) -- see git history for the comparison.
"""

from __future__ import annotations

import os

from google import genai

# Free-tier-eligible Flash models (see .env.example / project README for
# where to get a key: https://aistudio.google.com/apikey).
#
# Deliberately NOT the newest "gemini-3.8-flash" -- hit its free-tier
# quota live (confirmed via a real 429): only ~20 requests/day. gemini-2.5-
# flash is the previous Flash generation and gets a far more generous
# free-tier allowance (hundreds to 1500+ requests/day depending on when
# Google's own docs were last updated) while still supporting the same
# audio-in/audio-out interactions.create() calls this project relies on.
# Given development alone burns through single-digit-to-dozens of calls
# per session, 20/day is not workable; recheck current per-model limits
# at https://ai.google.dev/gemini-api/docs/rate-limits before changing
# this again.
TEXT_MODEL = "gemini-2.5-flash"
TTS_MODEL = "gemini-2.5-flash-preview-tts"

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
                "key from https://aistudio.google.com/apikey (no credit card needed)."
            )
        _client = genai.Client(api_key=api_key)
    return _client
