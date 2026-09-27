"""Structured visual observations -- the vision-side counterpart to
Action in models.py: a strict, validated boundary the model reasons over,
never a raw frame or an unvalidated blob of text.

Gemini's generate_content API is constrained by a JSON response schema
and grounds normalized image coordinates accurately
against known object positions in a test image (within ~1% of the
computed true center). That's what DetectedObject.image_x/image_y rely
on -- see vision.py.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ObservationPurpose = Literal[
    "scene",  # ambient/periodic re-observation while engaged
    "object_memory",  # "remember this object" -- about to call remember_object
    "goal_planning",  # about to act toward a spoken goal
    "goal_verification",  # after the goal's final action, before finish_goal
]


class DetectedObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=80)
    attributes: str = Field(default="", max_length=200)
    image_x: float = Field(ge=0.0, le=1.0)
    image_y: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    color: str | None = Field(default=None, max_length=40)
    notes: str | None = Field(default=None, max_length=200)


class Observation(BaseModel):
    """One capture-and-describe result. observation_id/captured_at_monotonic
    let ObservationRegistry enforce freshness and let goal actions refuse
    to act on a stale or unknown observation (see GoalCoordinator)."""

    model_config = ConfigDict(extra="forbid")

    observation_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    captured_at_monotonic: float = Field(default_factory=time.monotonic)
    purpose: ObservationPurpose
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    objects: list[DetectedObject] = Field(default_factory=list)

    def age_s(self, now_monotonic: float | None = None) -> float:
        now = now_monotonic if now_monotonic is not None else time.monotonic()
        return max(0.0, now - self.captured_at_monotonic)
