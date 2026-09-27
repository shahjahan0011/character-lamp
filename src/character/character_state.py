"""Explicit character states, wrapping the flags CharacterOrchestrator
already tracked (engaged, turn-in-flight, awaiting-first-audio, active
goal stage) rather than rewriting its control flow. A stored, validated
state (not just a computed property) is what makes "the state controls
light/motion/mic gating/error recovery" a checkable claim instead of an
implicit side effect of several booleans agreeing by convention.
"""

from __future__ import annotations

from enum import StrEnum


class CharacterState(StrEnum):
    DORMANT = "dormant"
    NOTICING = "noticing"
    ENGAGED = "engaged"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ACTING = "acting"
    VERIFYING = "verifying"
    DISENGAGING = "disengaging"
    ERROR = "error"


class InvalidTransitionError(ValueError):
    pass


ALLOWED_TRANSITIONS: dict[CharacterState, frozenset[CharacterState]] = {
    CharacterState.DORMANT: frozenset({CharacterState.NOTICING, CharacterState.ERROR}),
    CharacterState.NOTICING: frozenset({CharacterState.ENGAGED, CharacterState.DISENGAGING, CharacterState.ERROR}),
    CharacterState.ENGAGED: frozenset({
        CharacterState.LISTENING, CharacterState.THINKING, CharacterState.SPEAKING,
        CharacterState.ACTING, CharacterState.DISENGAGING, CharacterState.ERROR,
    }),
    CharacterState.LISTENING: frozenset({
        CharacterState.ENGAGED, CharacterState.THINKING, CharacterState.DISENGAGING, CharacterState.ERROR,
    }),
    CharacterState.THINKING: frozenset({
        CharacterState.SPEAKING, CharacterState.ACTING, CharacterState.ENGAGED,
        CharacterState.DISENGAGING, CharacterState.ERROR,
    }),
    CharacterState.SPEAKING: frozenset({
        CharacterState.ENGAGED, CharacterState.LISTENING, CharacterState.ACTING,
        CharacterState.DISENGAGING, CharacterState.ERROR,
    }),
    CharacterState.ACTING: frozenset({CharacterState.VERIFYING, CharacterState.ENGAGED, CharacterState.ERROR}),
    CharacterState.VERIFYING: frozenset({CharacterState.SPEAKING, CharacterState.ENGAGED, CharacterState.ERROR}),
    CharacterState.DISENGAGING: frozenset({CharacterState.DORMANT, CharacterState.ERROR}),
    CharacterState.ERROR: frozenset({CharacterState.DORMANT, CharacterState.ENGAGED}),
}


class CharacterStateMachine:
    def __init__(self, on_debug=None):
        self._state = CharacterState.DORMANT
        self._on_debug = on_debug or (lambda msg: None)

    @property
    def state(self) -> CharacterState:
        return self._state

    def transition(self, target: CharacterState) -> bool:
        """Returns whether the transition actually happened. A no-op
        (same state) is fine and reported as "no change"; a genuinely
        disallowed edge raises rather than silently corrupting state --
        callers that shouldn't ever hit an invalid edge (e.g. tick()'s
        own well-ordered logic) are expected to let this propagate during
        development, not swallow it."""
        if target == self._state:
            return False
        if target not in ALLOWED_TRANSITIONS[self._state]:
            raise InvalidTransitionError(f"illegal transition {self._state.value} -> {target.value}")
        self._on_debug(f"state: {self._state.value} -> {target.value}")
        self._state = target
        return True
