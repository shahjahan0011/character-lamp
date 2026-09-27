"""In-process scene memory: what has the lamp observed recently.

Deliberately simple -- a short list of (label, attributes), no persistence
across restarts, no embedding/vector search. The challenge asks for
"retains useful information... later answers a spoken question about it"
within one continuous demo, not a durable database; a plain list handed
to the model as context and answered in natural language covers that
without building infrastructure the demo doesn't need.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.speech.vision import ObservedObject


@dataclass
class MemoryEntry:
    label: str
    attributes: str
    observed_at: float = field(default_factory=time.time)


class SceneMemory:
    def __init__(self, max_entries: int = 20):
        self._entries: list[MemoryEntry] = []
        self._max_entries = max_entries

    def remember(self, objects: list[ObservedObject]) -> bool:
        """Merges freshly observed objects into memory, deduped by label
        (case-insensitive) rather than blindly appended -- with periodic
        re-observation (see CharacterOrchestrator), the same persistent
        object (a monitor that's always there) would otherwise pile up a
        new duplicate entry every scan instead of just refreshing it.

        Returns whether anything actually changed (a new label appeared,
        or an existing one's attributes changed) so a caller can decide
        whether it's worth telling the model about -- re-sending an
        identical "here's what you've seen" nudge every scan would just
        bloat the live conversation's context for no new information."""
        now = time.time()
        changed = False
        by_label = {e.label.lower(): e for e in self._entries}
        for obj in objects:
            key = obj.label.lower()
            existing = by_label.get(key)
            if existing is None:
                entry = MemoryEntry(label=obj.label, attributes=obj.attributes, observed_at=now)
                self._entries.append(entry)
                by_label[key] = entry
                changed = True
            else:
                if existing.attributes != obj.attributes:
                    changed = True
                existing.attributes = obj.attributes
                existing.observed_at = now
        self._entries = self._entries[-self._max_entries :]
        return changed

    def as_context_text(self) -> str:
        """Handed to the dialogue model as plain context -- it does the
        "does this answer their question" reasoning, not us; no query
        matching/parsing logic to get wrong here.

        Explicitly tells the model to use the list, and by name -- without
        this instruction, tested live, the model tends to answer a literal
        "what's on my desk" with "I haven't seen anything" even when the
        list is non-empty, apparently reasoning that kitchen items (say)
        aren't technically "on a desk" rather than treating the list as
        "things I've noticed nearby" in the spirit the question was asked."""
        if not self._entries:
            return "You have not noticed any objects nearby yet."
        lines = [f"- {e.label}: {e.attributes}" for e in self._entries]
        return (
            "Specific objects you have personally noticed nearby (use these "
            "by name if asked what you've seen/noticed -- treat this as "
            "everything currently in view, don't say you haven't seen "
            "anything if this list is non-empty):\n" + "\n".join(lines)
        )

    def __len__(self) -> int:
        return len(self._entries)
