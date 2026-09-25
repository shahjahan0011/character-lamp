"""Single-turn spoken dialogue via Gemini.

Deliberately minimal for now: no scene memory, no goal-directed planning,
no multi-turn conversation history -- those are separate, later pieces
(demo moments 4 and 5). This module answers only "what was just said",
which is what demo moment 3 (spoken interaction) actually asks for.
"""

from __future__ import annotations

from .gemini_client import TEXT_MODEL, get_client

PERSONA = (
    "You are the voice of a small friendly desk lamp character (like "
    "Pixar's Luxo Jr., but able to talk). Someone just spoke to you. "
    "Reply warmly and conversationally in 1-2 short sentences -- this "
    "will be spoken aloud, not read, so keep it natural to say out loud."
)


def respond(transcript: str) -> str:
    client = get_client()
    interaction = client.interactions.create(
        model=TEXT_MODEL,
        input=[{"type": "text", "text": f"{PERSONA}\n\nThey said: \"{transcript}\""}],
    )
    return (interaction.output_text or "").strip()
