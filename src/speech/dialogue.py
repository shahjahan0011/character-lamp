"""Single-turn spoken dialogue via Gemini.

Deliberately minimal for now: no scene memory, no goal-directed planning,
no multi-turn conversation history -- those are separate, later pieces
(demo moments 4 and 5). This module answers only "what was just said",
which is what demo moment 3 (spoken interaction) actually asks for.

Also picks one physical reaction to go with the reply (nod/shake_head/
excited/curious/none), the same idea as a reference implementation the
project owner pointed at (a Gemini Live session where the model calls a
`perform_gesture` tool as part of its turn). We're not on the Live API
here -- this is the simpler request/response pipeline -- so instead of a
real tool call, the model is asked to emit both fields in one plain-text
response, parsed with a regex. Less elegant, but it's one predictable
extra line in a call we already know works, rather than a new API shape
to get subtly wrong under time pressure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .gemini_client import TEXT_MODEL, get_client

Gesture = Literal["nod", "shake_head", "excited", "curious", "none"]
_VALID_GESTURES = {"nod", "shake_head", "excited", "curious", "none"}

PERSONA = (
    "You are the voice of a small friendly desk lamp character (like "
    "Pixar's Luxo Jr., but able to talk). Someone just spoke to you. "
    "Reply warmly and conversationally in 1-2 short sentences -- this "
    "will be spoken aloud, not read, so keep it natural to say out loud.\n\n"
    "You can also physically react. Pick exactly one reaction that best "
    "matches your reply's emotional tone:\n"
    "- nod: agreeing, affirming, confirming something\n"
    "- shake_head: disagreeing, saying no, correcting a mistaken assumption\n"
    "- excited: delighted, enthusiastic, celebrating something\n"
    "- curious: intrigued, asking a question back, puzzling over something\n"
    "- none: neutral, none of the above fit\n\n"
    "Respond in EXACTLY this two-line format, nothing else, no markdown:\n"
    "GESTURE: <nod|shake_head|excited|curious|none>\n"
    "REPLY: <your spoken reply>"
)

_GESTURE_RE = re.compile(r"GESTURE:\s*(\w+)", re.IGNORECASE)
_REPLY_RE = re.compile(r"REPLY:\s*(.+)", re.IGNORECASE | re.DOTALL)


@dataclass
class DialogueResponse:
    reply: str
    gesture: Gesture


def _parse(raw: str) -> DialogueResponse:
    gesture: Gesture = "none"
    reply = raw.strip()

    gesture_match = _GESTURE_RE.search(raw)
    if gesture_match and gesture_match.group(1).lower() in _VALID_GESTURES:
        gesture = gesture_match.group(1).lower()  # type: ignore[assignment]

    reply_match = _REPLY_RE.search(raw)
    if reply_match:
        reply = reply_match.group(1).strip()

    return DialogueResponse(reply=reply, gesture=gesture)


def respond(transcript: str, timeout_s: float = 30.0) -> DialogueResponse:
    client = get_client()
    interaction = client.interactions.create(
        model=TEXT_MODEL,
        input=[{"type": "text", "text": f'{PERSONA}\n\nThey said: "{transcript}"'}],
        timeout=timeout_s,
    )
    return _parse(interaction.output_text or "")
