"""Single-turn spoken dialogue via Gemini.

Deliberately minimal for now: no scene memory, no goal-directed planning,
no multi-turn conversation history -- those are separate, later pieces
(demo moments 4 and 5). This module answers only "what was just said",
which is what demo moment 3 (spoken interaction) actually asks for.

respond_to_audio() sends the recorded utterance directly -- no separate
stt.transcribe() call first. Measured live: a dedicated transcribe-only
call plus this reply call cost roughly 34s + a few seconds, back to back,
for no benefit -- audio understanding and the reply both come from the
same model call anyway, so asking for the transcript *and* the reply in
one request removes one full network round-trip. stt.transcribe() is
kept for callers that only want a transcript (e.g. scene-memory question
matching later), but the spoken-interaction path no longer uses it.

Also picks one physical reaction to go with the reply (nod/shake_head/
excited/curious/none), the same idea as a reference implementation the
project owner pointed at (a Gemini Live session where the model calls a
`perform_gesture` tool as part of its turn). We're not on the Live API
here -- this is the simpler request/response pipeline -- so instead of a
real tool call, the model is asked to emit all fields in one plain-text
response, parsed with a regex. Less elegant, but it's predictable extra
lines in a call we already know works, rather than a new API shape to get
subtly wrong under time pressure.

Requests stream=True. Measured live: the same call, non-streaming vs.
streaming, was 23-60+s vs. 7.87s -- not a small tweak, apparently a
genuinely different and much faster server-side code path, even though
in practice it still arrives as 1-2 text deltas rather than many
incremental ones. Combined with tts.py's streaming (16s -> 3.8s), total
round-trip latency drops from 40-90s+ down to roughly 12-15s. timeout_s
is still set with real margin above the fast case -- free-tier latency
is variable, not just slow, and a slow-but-successful call beats a
needlessly aborted one. DialogueWorker surfaces a real timeout as a
visible "something went wrong" cue rather than silence; see its _failed
flag.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Literal

from .gemini_client import TEXT_MODEL, get_client

Gesture = Literal["nod", "shake_head", "excited", "curious", "none"]
_VALID_GESTURES = {"nod", "shake_head", "excited", "curious", "none"}

_REACTION_RULES = (
    "You can also physically react. Pick exactly one reaction that best "
    "matches your reply's emotional tone:\n"
    "- nod: agreeing, affirming, confirming something\n"
    "- shake_head: disagreeing, saying no, correcting a mistaken assumption\n"
    "- excited: delighted, enthusiastic, celebrating something\n"
    "- curious: intrigued, asking a question back, puzzling over something\n"
    "- none: neutral, none of the above fit\n\n"
)

PERSONA = (
    "You are the voice of a small friendly desk lamp character (like "
    "Pixar's Luxo Jr., but able to talk). Someone just spoke to you. "
    "Reply warmly and conversationally in 1-2 short sentences -- this "
    "will be spoken aloud, not read, so keep it natural to say out loud.\n\n"
    + _REACTION_RULES
    + "Respond in EXACTLY this two-line format, nothing else, no markdown:\n"
    "GESTURE: <nod|shake_head|excited|curious|none>\n"
    "REPLY: <your spoken reply>"
)

AUDIO_PERSONA = (
    "You are the voice of a small friendly desk lamp character (like "
    "Pixar's Luxo Jr., but able to talk). Someone just spoke to you in "
    "this audio clip. Listen to what they said and reply warmly and "
    "conversationally in 1-2 short sentences -- this will be spoken "
    "aloud, not read, so keep it natural to say out loud.\n\n"
    + _REACTION_RULES
    + "Respond in EXACTLY this three-line format, nothing else, no markdown:\n"
    "TRANSCRIPT: <verbatim transcript of what they said, or an empty "
    "line if there was no speech>\n"
    "GESTURE: <nod|shake_head|excited|curious|none>\n"
    "REPLY: <your spoken reply>"
)

_GESTURE_RE = re.compile(r"GESTURE:\s*(\w+)", re.IGNORECASE)
_REPLY_RE = re.compile(r"REPLY:\s*(.+)", re.IGNORECASE | re.DOTALL)
_TRANSCRIPT_RE = re.compile(r"TRANSCRIPT:\s*(.*)", re.IGNORECASE)


@dataclass
class DialogueResponse:
    reply: str
    gesture: Gesture


@dataclass
class AudioDialogueResponse:
    transcript: str
    reply: str
    gesture: Gesture


def _extract_gesture(raw: str) -> Gesture:
    match = _GESTURE_RE.search(raw)
    if match and match.group(1).lower() in _VALID_GESTURES:
        return match.group(1).lower()  # type: ignore[return-value]
    return "none"


def _extract_reply(raw: str) -> str:
    match = _REPLY_RE.search(raw)
    return match.group(1).strip() if match else raw.strip()


def _parse(raw: str) -> DialogueResponse:
    return DialogueResponse(reply=_extract_reply(raw), gesture=_extract_gesture(raw))


def _parse_audio(raw: str) -> AudioDialogueResponse:
    transcript_match = _TRANSCRIPT_RE.search(raw)
    transcript = transcript_match.group(1).strip() if transcript_match else ""
    return AudioDialogueResponse(
        transcript=transcript, reply=_extract_reply(raw), gesture=_extract_gesture(raw)
    )


def _collect_text_stream(stream) -> str:
    chunks: list[str] = []
    for event in stream:
        delta = getattr(event, "delta", None)
        if delta is not None and getattr(delta, "type", None) == "text":
            chunks.append(delta.text or "")
    return "".join(chunks)


def respond(transcript: str, timeout_s: float = 30.0) -> DialogueResponse:
    client = get_client()
    stream = client.interactions.create(
        model=TEXT_MODEL,
        input=[{"type": "text", "text": f'{PERSONA}\n\nThey said: "{transcript}"'}],
        stream=True,
        timeout=timeout_s,
    )
    return _parse(_collect_text_stream(stream))


def respond_to_audio(wav_bytes: bytes, timeout_s: float = 45.0) -> AudioDialogueResponse:
    client = get_client()
    stream = client.interactions.create(
        model=TEXT_MODEL,
        input=[
            {"type": "text", "text": AUDIO_PERSONA},
            {
                "type": "audio",
                "data": base64.b64encode(wav_bytes).decode("utf-8"),
                "mime_type": "audio/wav",
            },
        ],
        stream=True,
        timeout=timeout_s,
    )
    return _parse_audio(_collect_text_stream(stream))
