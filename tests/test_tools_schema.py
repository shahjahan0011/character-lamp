"""Strict tool-argument schemas: unknown fields and bounds are rejected."""

import pytest
from pydantic import ValidationError

from src.protocol.tools import (
    TOOL_ARG_MODELS,
    FinishGoalArgs,
    LookAtImagePointArgs,
    PerformGestureArgs,
    RecallMemoryArgs,
    RememberObjectArgs,
    RequestObservationArgs,
    SetLightToolArgs,
)


def test_all_declared_tools_have_a_schema():
    expected = {
        "request_observation",
        "recall_memory",
        "remember_object",
        "look_at_image_point",
        "set_light",
        "perform_gesture",
        "finish_goal",
    }
    assert set(TOOL_ARG_MODELS.keys()) == expected


def test_request_observation_rejects_unknown_purpose():
    with pytest.raises(ValidationError):
        RequestObservationArgs(purpose="not_a_real_purpose")


def test_request_observation_accepts_each_valid_purpose():
    for purpose in ("scene", "object_memory", "goal_planning", "goal_verification"):
        RequestObservationArgs(purpose=purpose)


def test_recall_memory_enforces_limit_bounds():
    RecallMemoryArgs(query="mug", limit=5)
    with pytest.raises(ValidationError):
        RecallMemoryArgs(query="mug", limit=0)
    with pytest.raises(ValidationError):
        RecallMemoryArgs(query="mug", limit=100)


def test_recall_memory_rejects_empty_query():
    with pytest.raises(ValidationError):
        RecallMemoryArgs(query="", limit=5)


def test_remember_object_rejects_out_of_range_coordinates():
    with pytest.raises(ValidationError):
        RememberObjectArgs(
            observation_id="abc", label="mug", color=None, attributes="", image_x=1.5, image_y=0.5,
            confidence=0.5, notes=None,
        )


def test_remember_object_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        RememberObjectArgs(
            observation_id="abc", label="mug", color=None, attributes="", image_x=0.5, image_y=0.5,
            confidence=0.5, notes=None, joint_angle=1.0,
        )


def test_look_at_image_point_enforces_bounds():
    LookAtImagePointArgs(observation_id="abc", x=0.0, y=1.0)
    with pytest.raises(ValidationError):
        LookAtImagePointArgs(observation_id="abc", x=-0.01, y=0.5)
    with pytest.raises(ValidationError):
        LookAtImagePointArgs(observation_id="abc", x=0.5, y=1.01)


def test_look_at_image_point_cannot_express_joint_angles():
    """The whole point of this schema: no field exists for a joint name,
    angle, or torque -- only a normalized 2-D image point."""
    fields = set(LookAtImagePointArgs.model_fields.keys())
    assert fields == {"observation_id", "x", "y"}


def test_set_light_rejects_invalid_mode():
    with pytest.raises(ValidationError):
        SetLightToolArgs(brightness=0.5, mode="strobe", duration_ms=100)


def test_set_light_enforces_duration_bounds():
    SetLightToolArgs(brightness=0.5, mode="steady", duration_ms=0)
    with pytest.raises(ValidationError):
        SetLightToolArgs(brightness=0.5, mode="steady", duration_ms=10_000)


def test_perform_gesture_rejects_unknown_gesture_name():
    with pytest.raises(ValidationError):
        PerformGestureArgs(name="backflip")


def test_perform_gesture_enforces_intensity_bounds():
    PerformGestureArgs(name="nod", intensity=0.25)
    with pytest.raises(ValidationError):
        PerformGestureArgs(name="nod", intensity=0.1)
    with pytest.raises(ValidationError):
        PerformGestureArgs(name="nod", intensity=1.5)


def test_finish_goal_requires_all_fields():
    FinishGoalArgs(
        goal_id="g1", post_action_observation_id="obs1", success=True, evidence="visible", confidence=0.9
    )
    with pytest.raises(ValidationError):
        FinishGoalArgs(goal_id="g1", success=True, evidence="visible", confidence=0.9)  # missing observation id


def test_no_tool_schema_exposes_a_file_path_or_shell_field():
    """A blunt but important guard: none of these argument models should
    ever grow a field that could carry a file path, shell command, or
    arbitrary code -- that's the whole safety boundary these tools exist
    to enforce."""
    forbidden_names = {"path", "file", "command", "cmd", "shell", "code", "script", "exec"}
    for name, model in TOOL_ARG_MODELS.items():
        field_names = {f.lower() for f in model.model_fields}
        overlap = field_names & forbidden_names
        assert not overlap, f"{name} exposes a suspicious field: {overlap}"
