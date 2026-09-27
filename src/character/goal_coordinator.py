"""Local enforcement of the goal-directed-action workflow (demo moment 5).

This is deliberately the *only* place "did the goal actually earn success"
gets decided. Gemini can say anything it wants through finish_goal's
`success`/`evidence`/`confidence` arguments, but GoalCoordinator won't let
a goal reach VERIFIED unless the real local invariants were actually
satisfied first -- a fresh goal_planning observation before acting, at
least one completed action, and a goal_verification observation captured
strictly after that action completed. Every violation raises GoalError,
which ToolGateway turns into a structured tool error (never an unhandled
exception, never a silently-ignored one) so a Live turn can't hang and the
model gets an explicit reason to adapt rather than a mysterious failure.

Uses time.monotonic() throughout for the same reason ObservationRegistry
does: ordering/deadline logic must never be corrupted by a wall-clock
jump.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

from src.character.observation_registry import ObservationRegistry

GoalStage = Literal["created", "observing", "planning", "acting", "reobserving", "verified", "failed"]


class GoalError(ValueError):
    """An invalid tool-call ordering against the goal workflow -- caught
    by ToolGateway and returned as a structured {"ok": False, "error":...}
    tool result rather than propagating."""


@dataclass
class Goal:
    goal_id: str
    stage: GoalStage = "created"
    planning_observation_id: str | None = None
    actions_completed: int = 0
    last_action_completed_at: float | None = None  # monotonic
    verification_observation_id: str | None = None
    retries_used: int = 0
    created_at: float = field(default_factory=time.monotonic)


class GoalCoordinator:
    def __init__(self, registry: ObservationRegistry, max_retries: int = 1):
        self._registry = registry
        self._max_retries = max_retries
        self._lock = threading.Lock()
        self._goals: dict[str, Goal] = {}

    def start_goal(self) -> Goal:
        goal = Goal(goal_id=uuid.uuid4().hex, stage="observing")
        with self._lock:
            self._goals[goal.goal_id] = goal
        return goal

    def get(self, goal_id: str) -> Goal | None:
        with self._lock:
            return self._goals.get(goal_id)

    def _require(self, goal_id: str) -> Goal:
        goal = self.get(goal_id)
        if goal is None:
            raise GoalError(f"unknown goal_id {goal_id!r} -- start a new goal first")
        return goal

    def record_planning_observation(self, goal_id: str, observation_id: str) -> Goal:
        """A fresh goal_planning observation must exist before any
        scene-directed action -- this is what makes that a real,
        checkable requirement rather than a prompt-level request."""
        goal = self._require(goal_id)
        obs = self._registry.get(observation_id)
        if obs is None or not self._registry.is_fresh(observation_id):
            raise GoalError(f"observation {observation_id!r} is unknown or stale; request a fresh observation")
        if obs.purpose != "goal_planning":
            raise GoalError(f"observation {observation_id!r} was not captured for goal_planning")
        with self._lock:
            goal.planning_observation_id = observation_id
            goal.stage = "planning"
        return goal

    def record_action_completed(self, goal_id: str) -> Goal:
        goal = self._require(goal_id)
        if goal.planning_observation_id is None:
            raise GoalError("cannot act toward this goal before a fresh goal_planning observation is recorded")
        with self._lock:
            goal.stage = "acting"
            goal.actions_completed += 1
            goal.last_action_completed_at = time.monotonic()
        return goal

    def record_verification_observation(self, goal_id: str, observation_id: str) -> Goal:
        goal = self._require(goal_id)
        if goal.actions_completed < 1 or goal.last_action_completed_at is None:
            raise GoalError("cannot verify before at least one validated action has completed")
        obs = self._registry.get(observation_id)
        if obs is None or not self._registry.is_fresh(observation_id):
            raise GoalError(f"observation {observation_id!r} is unknown or stale")
        if obs.purpose != "goal_verification":
            raise GoalError(f"observation {observation_id!r} was not captured for goal_verification")
        if obs.captured_at_monotonic <= goal.last_action_completed_at:
            raise GoalError("the goal_verification observation must be captured after the final action completed")
        with self._lock:
            goal.verification_observation_id = observation_id
            goal.stage = "reobserving"
        return goal

    def finish(
        self,
        goal_id: str,
        post_action_observation_id: str,
        success: bool,
        evidence: str,
        confidence: float,
    ) -> Goal:
        goal = self._require(goal_id)
        if goal.verification_observation_id is None:
            raise GoalError("cannot finish before a fresh goal_verification observation is recorded")
        if post_action_observation_id != goal.verification_observation_id:
            raise GoalError(
                "finish_goal's post_action_observation_id must match the recorded goal_verification observation"
            )
        with self._lock:
            if success:
                goal.stage = "verified"
            elif goal.retries_used < self._max_retries:
                # One inconclusive retry: back to planning, not an outright
                # failure yet -- the model can request a fresh observation
                # and try again before this goal is given up on.
                goal.retries_used += 1
                goal.stage = "planning"
            else:
                goal.stage = "failed"
        return goal
