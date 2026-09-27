"""Strict, closed-schema arguments for every Gemini Live tool this project
exposes -- the model-facing counterpart to Action/Observation. Every model
here forbids unknown fields and bounds every numeric argument; none of them
can express a joint name, a joint angle, torque, a file path, a shell
command, or arbitrary code. Gestures/pointing/light go through the exact
same Action/ActionExecutor boundary as everything else -- these are just
the argument shapes the model is allowed to fill in.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .models import GestureName
from .observation import ObservationPurpose


class RequestObservationArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    purpose: ObservationPurpose


class RecallMemoryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=5, ge=1, le=20)


class RememberObjectArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation_id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=80)
    color: str | None = Field(default=None, max_length=40)
    attributes: str = Field(default="", max_length=200)
    image_x: float = Field(ge=0.0, le=1.0)
    image_y: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str | None = Field(default=None, max_length=200)


class LookAtImagePointArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observation_id: str = Field(min_length=1, max_length=64)
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class SetLightToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    brightness: float = Field(ge=0.0, le=1.0)
    mode: str = Field(default="steady", pattern="^(steady|pulse|ramp)$")
    duration_ms: int = Field(default=500, ge=0, le=5000)


class PerformGestureArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: GestureName
    intensity: float = Field(default=1.0, ge=0.25, le=1.0)


class FinishGoalArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal_id: str = Field(min_length=1, max_length=64)
    post_action_observation_id: str = Field(min_length=1, max_length=64)
    success: bool
    evidence: str = Field(min_length=1, max_length=300)
    confidence: float = Field(ge=0.0, le=1.0)


TOOL_ARG_MODELS: dict[str, type[BaseModel]] = {
    "request_observation": RequestObservationArgs,
    "recall_memory": RecallMemoryArgs,
    "remember_object": RememberObjectArgs,
    "look_at_image_point": LookAtImagePointArgs,
    "set_light": SetLightToolArgs,
    "perform_gesture": PerformGestureArgs,
    "finish_goal": FinishGoalArgs,
}
