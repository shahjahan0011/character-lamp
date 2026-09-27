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
import time
from collections.abc import Callable
from typing import TypeVar

from google import genai

# Free-tier-eligible Flash models (see .env.example / project README for
# where to get a key: https://aistudio.google.com/apikey). All three are
# configurable via environment variables so a different account/tier can
# swap models without a code change; the literals below are only the
# defaults, each chosen from a real, live-tested measurement, not from
# documentation alone (Google's own docs were stale for at least one of
# these -- see below).
#
# Superseded gemini-2.5-flash after confirming LIVE (not from docs, which
# turned out stale) that it's capped at only 20 requests/day free tier --
# a real rate_limit_exceeded hit repeatedly during a session that also
# re-observes the scene every ~15s while engaged, exhausting the day's
# quota within minutes. gemini-3.5-flash-lite tested clean (correct scene
# description, no error event) against the same account/key with quota to
# spare, and is Google's own currently-recommended model -- the older
# gemini-2.5-flash-lite is already a 404 for new users ("no longer
# available... use gemini-3.5-flash-lite").
VISION_MODEL = os.environ.get("GEMINI_VISION_MODEL", "gemini-3.5-flash-lite")

# NOT gemini-2.5-flash-preview-tts -- confirmed live via a real
# rate_limit_exceeded error (only surfaced after fixing error-swallowing;
# see GeminiStreamError below) that it's capped at 10 requests/day and 3
# requests/minute on the free tier. Every single spoken turn needs one
# TTS call, so 10/day makes any real testing or demo session go silent
# after a handful of replies -- this is almost certainly why TTS kept
# failing in ways that looked like unrelated bugs (timeouts, an
# "invalid_request" on short text, a generic "api_error") before the
# error-swallowing fix made the real rate_limit_exceeded message visible.
# gemini-3.8-flash-lite-tts tested clean (real audio deltas, no error
# event) against the same account/key.
TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-3.8-flash-lite-tts")

_client: genai.Client | None = None


class GeminiStreamError(RuntimeError):
    """A stream=True interactions.create() response carried an SSE
    ErrorEvent (event_type == "error") instead of -- or in addition to --
    real content.

    Confirmed live by reading the SDK source (types/interactions/
    errorevent.py) and reproducing it against the real API: a quota/rate-
    limit/invalid-request/server error surfaces as a normal event in the
    stream, not a raised exception. A collector that only looks at
    `.delta` (the shape used everywhere in this project) sees no deltas
    and just returns "" -- indistinguishable from "no speech in the clip"
    or "no objects in view" from the caller's side. Making this a real
    exception is what lets call_with_retry, and the caller's own
    telemetry, actually see what happened."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"Gemini stream error [{code}]: {message}")


# Confirmed retryable live: transient/server-side, not "asking again with
# the same input would fail the same way".
_RETRYABLE_ERROR_CODES = {
    "rate_limit_exceeded",
    "too_many_requests",
    "internal_error",
    "api_error",
    "server_error",
    "unavailable",
}
# Deliberately NOT retried: quota_exceeded won't clear up within a few
# seconds (it's a daily cap), and invalid_request/auth errors are
# deterministic for the same input -- retrying just burns another call.
_NON_RETRYABLE_ERROR_CODES = {
    "quota_exceeded",
    "invalid_request",
    "authentication_error",
    "permission_denied",
}


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


def collect_text_stream(stream) -> str:
    """Accumulates a stream=True interactions.create() response into a
    single string. Shared by dialogue.py and vision.py -- streaming is
    used everywhere now, not for incremental chunking (short responses
    still arrive as 1-2 deltas) but because it measured dramatically
    faster server-side (see dialogue.py's docstring for the numbers)."""
    chunks: list[str] = []
    for event in stream:
        raise_if_error_event(event)
        delta = getattr(event, "delta", None)
        if delta is not None and getattr(delta, "type", None) == "text":
            chunks.append(delta.text or "")
    return "".join(chunks)


def raise_if_error_event(event) -> None:
    """Raises GeminiStreamError if this stream event is an SSE ErrorEvent.
    Shared by collect_text_stream (text/vision) and tts.py's own delta
    loop (audio) -- both need the same check, since ErrorEvent can arrive
    on any interactions.create(stream=True) response regardless of what
    kind of content it's otherwise producing."""
    if getattr(event, "event_type", None) != "error":
        return
    err = getattr(event, "error", None)
    code = getattr(err, "code", None) or "unknown"
    message = getattr(err, "message", None) or str(err)
    raise GeminiStreamError(code, message)


_T = TypeVar("_T")


def call_with_retry(
    fn: Callable[[], _T],
    retries: int = 1,
    backoff_s: float = 1.0,
    on_debug: Callable[[str], None] | None = None,
) -> _T:
    """Runs fn() and retries on a client-side timeout.

    Confirmed live: free-tier latency doesn't just run slow, it occasionally
    spikes well past an already-generous per-call timeout (30-45s) even
    though the same call typically finishes in 10-20s -- three back-to-back
    calls (observe, reply, TTS) around one engagement makes this worse. A
    single retry turns "this whole turn silently produced nothing" into
    "this turn was a few seconds slower" for exactly that failure mode.
    Also retries a GeminiStreamError carrying a transient error code (rate
    limit, internal/server error) -- confirmed live these happen even on
    a request that would otherwise succeed. Anything else (a timeout-
    shaped exception aside) is re-raised immediately: quota_exceeded,
    invalid_request, and auth errors are deterministic for the same input,
    so retrying just burns another call without changing the outcome.
    """
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:
            if not _is_retryable(exc) or attempt == retries:
                raise
            last_exc = exc
            if on_debug:
                on_debug(f"  (Gemini call failed, retrying: {exc})")
            time.sleep(backoff_s)
    assert last_exc is not None
    raise last_exc


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, GeminiStreamError):
        return exc.code in _RETRYABLE_ERROR_CODES
    return "timeout" in str(exc).lower() or "timeout" in type(exc).__name__.lower()
