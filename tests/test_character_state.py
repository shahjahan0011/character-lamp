"""CharacterState transition validation."""

import pytest

from src.character.character_state import (
    CharacterState,
    CharacterStateMachine,
    InvalidTransitionError,
)


def test_starts_dormant():
    machine = CharacterStateMachine()
    assert machine.state == CharacterState.DORMANT


def test_valid_transition_sequence():
    machine = CharacterStateMachine()
    assert machine.transition(CharacterState.NOTICING) is True
    assert machine.transition(CharacterState.ENGAGED) is True
    assert machine.transition(CharacterState.LISTENING) is True
    assert machine.transition(CharacterState.THINKING) is True
    assert machine.transition(CharacterState.SPEAKING) is True
    assert machine.transition(CharacterState.ACTING) is True
    assert machine.transition(CharacterState.VERIFYING) is True
    assert machine.transition(CharacterState.ENGAGED) is True
    assert machine.transition(CharacterState.DISENGAGING) is True
    assert machine.transition(CharacterState.DORMANT) is True


def test_same_state_transition_is_a_noop_not_an_error():
    machine = CharacterStateMachine()
    assert machine.transition(CharacterState.DORMANT) is False
    assert machine.state == CharacterState.DORMANT


def test_illegal_transition_raises():
    machine = CharacterStateMachine()
    with pytest.raises(InvalidTransitionError):
        machine.transition(CharacterState.SPEAKING)  # can't go straight from DORMANT to SPEAKING


def test_error_reachable_from_every_non_error_state():
    from src.character.character_state import ALLOWED_TRANSITIONS

    for state, allowed in ALLOWED_TRANSITIONS.items():
        if state is CharacterState.ERROR:
            continue
        assert CharacterState.ERROR in allowed, f"{state} cannot reach ERROR"


def test_dormant_reachable_from_error_and_disengaging():
    from src.character.character_state import ALLOWED_TRANSITIONS

    assert CharacterState.DORMANT in ALLOWED_TRANSITIONS[CharacterState.ERROR]
    assert CharacterState.DORMANT in ALLOWED_TRANSITIONS[CharacterState.DISENGAGING]
