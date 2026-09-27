"""Executes every Gemini Live tool call against the strict argument
schemas in protocol/tools.py and the local invariants owned by
GoalCoordinator/ObservationRegistry/SceneMemory. This is the one place a
tool call turns into a real side effect (a physical Action, a memory
write, a goal-stage transition) -- nothing upstream of here (live_client's
event plumbing, fsm.py's tick loop) knows what any tool actually does.

Every call returns a structured {"ok": bool, ...} result rather than
raising -- most stateful Live tools are declared Behavior.BLOCKING (the
purely expressive gesture is non-blocking; see live_client.py), so an
unhandled exception here would hang that turn rather than just failing
one call. request_observation is the one exception to "returns
synchronously": it needs a real network vision call, which happens on
SceneObserver's background thread, so execute() queues it and returns
None -- CharacterOrchestrator completes that specific tool call later via
SceneObserver's own polling (see fsm.py).

Single-current-goal model: only one goal is tracked as "in flight" at a
time, matching the demo flow this challenge asks for (one spoken goal,
planned, acted on, verified, finished, before the next). A goal_planning
request_observation starts a new goal if none is active; its result
includes goal_id so the model can carry it forward into finish_goal.
Any look_at_image_point/perform_gesture/set_light call while that goal is
between "planning" and verified/failed counts as its validated action --
a deliberate simplification (a purely reactive gesture mid-goal would
also count) rather than requiring an explicit goal_id on every physical
tool, which the schema in protocol/tools.py doesn't carry.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from pydantic import ValidationError

from src.body.executor import ActionExecutor, image_point_to_pan_tilt
from src.character.goal_coordinator import GoalCoordinator, GoalError
from src.character.memory import SceneMemory
from src.character.observation_registry import ObservationRegistry
from src.character.scene_observer import SceneObserver
from src.protocol.models import Action
from src.protocol.tools import TOOL_ARG_MODELS

ACTIVE_GOAL_STAGES = frozenset({"observing", "planning", "acting", "reobserving"})


class ToolGateway:
    def __init__(
        self,
        executor: ActionExecutor,
        observer: SceneObserver,
        registry: ObservationRegistry,
        memory: SceneMemory,
        goals: GoalCoordinator,
        get_latest_frame: Callable[[], np.ndarray | None],
        on_debug: Callable[[str], None] | None = None,
    ):
        self._executor = executor
        self._observer = observer
        self._registry = registry
        self._memory = memory
        self._goals = goals
        self._get_latest_frame = get_latest_frame
        self._on_debug = on_debug or (lambda msg: None)
        self._current_goal_id: str | None = None

    def execute(self, name: str, args: dict, call_id: str) -> dict | None:
        """Returns a result dict, or None if the call was queued for async
        completion (request_observation only -- see module docstring)."""
        model_cls = TOOL_ARG_MODELS.get(name)
        if model_cls is None:
            return {"ok": False, "error": f"unknown tool {name!r}"}
        try:
            validated = model_cls.model_validate(args)
        except ValidationError as exc:
            return {"ok": False, "error": f"invalid arguments for {name}: {exc}"}

        try:
            if name == "request_observation":
                return self._request_observation(validated, call_id)
            if name == "recall_memory":
                return self._recall_memory(validated)
            if name == "remember_object":
                return self._remember_object(validated)
            if name == "look_at_image_point":
                return self._look_at_image_point(validated)
            if name == "set_light":
                return self._set_light(validated)
            if name == "perform_gesture":
                return self._perform_gesture(validated)
            if name == "finish_goal":
                return self._finish_goal(validated)
            return {"ok": False, "error": f"tool {name!r} has no execution path"}
        except GoalError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 -- a tool call must never hang or crash the session
            self._on_debug(f"tool_gateway: unexpected error executing {name}: {exc}")
            return {"ok": False, "error": "internal error executing this tool"}

    # -- individual tools ------------------------------------------------------

    def _request_observation(self, args, call_id: str) -> dict | None:
        frame = self._get_latest_frame()
        if frame is None:
            return {"ok": False, "error": "no camera frame currently available"}
        if args.purpose == "goal_planning" and (
            self._current_goal_id is None
            or (self._goals.get(self._current_goal_id) is not None
                and self._goals.get(self._current_goal_id).stage not in ACTIVE_GOAL_STAGES)
        ):
            self._current_goal_id = self._goals.start_goal().goal_id
        submitted = self._observer.submit_observation(
            frame, purpose=args.purpose, tool_call_id=call_id, tool_name="request_observation"
        )
        if not submitted:
            return {"ok": False, "error": "a previous observation is still in progress; try again shortly"}
        return None  # completes later -- see fsm.py's handling of SceneObserver's results

    def _recall_memory(self, args) -> dict:
        records = self._memory.recall(args.query, limit=args.limit)
        return {
            "ok": True,
            "records": [
                {
                    "label": r.label,
                    "color": r.color,
                    "attributes": r.attributes,
                    "image_x": r.image_x,
                    "image_y": r.image_y,
                    "confidence": r.confidence,
                    "notes": r.notes,
                }
                for r in records
            ],
        }

    def _remember_object(self, args) -> dict:
        if not self._registry.is_fresh(args.observation_id):
            return {"ok": False, "error": f"observation {args.observation_id!r} is unknown or stale"}
        record = self._memory.remember_one(
            label=args.label,
            color=args.color,
            attributes=args.attributes,
            image_x=args.image_x,
            image_y=args.image_y,
            confidence=args.confidence,
            source_observation_id=args.observation_id,
            notes=args.notes,
        )
        return {"ok": True, "label": record.label}

    def _look_at_image_point(self, args) -> dict:
        if not self._registry.is_fresh(args.observation_id):
            return {"ok": False, "error": f"observation {args.observation_id!r} is unknown or stale"}
        pan, tilt = image_point_to_pan_tilt(args.x, args.y)
        self._executor.run(Action(kind="point_at", params={"pan": pan, "tilt": tilt}))
        self._advance_goal_if_active()
        return {"ok": True, "pan": pan, "tilt": tilt}

    def _set_light(self, args) -> dict:
        self._executor.run(
            Action(kind="set_light", params={"on": True, "color": [1.0, 0.95, 0.76], "brightness": args.brightness})
        )
        self._advance_goal_if_active()
        return {"ok": True}

    def _perform_gesture(self, args) -> dict:
        self._executor.run(Action(kind=args.name, params={}))
        self._advance_goal_if_active()
        return {"ok": True}

    def _finish_goal(self, args) -> dict:
        goal = self._goals.finish(
            args.goal_id, args.post_action_observation_id, args.success, args.evidence, args.confidence
        )
        if goal.goal_id == self._current_goal_id and goal.stage in ("verified", "failed"):
            self._current_goal_id = None
        return {"ok": True, "stage": goal.stage, "retries_used": goal.retries_used}

    # -- helpers -----------------------------------------------------------------

    def _advance_goal_if_active(self) -> None:
        if self._current_goal_id is None:
            return
        goal = self._goals.get(self._current_goal_id)
        if goal is not None and goal.stage in ("planning", "acting"):
            self._goals.record_action_completed(self._current_goal_id)

    def complete_observation_for_goal(self, observation_id: str, purpose: str) -> None:
        """Called once a request_observation job finishes (see fsm.py) --
        advances the current goal's stage if this observation was for it.
        A no-op for purpose="scene"/"object_memory" or if no goal is
        active; those don't participate in the goal workflow."""
        if self._current_goal_id is None:
            return
        try:
            if purpose == "goal_planning":
                self._goals.record_planning_observation(self._current_goal_id, observation_id)
            elif purpose == "goal_verification":
                self._goals.record_verification_observation(self._current_goal_id, observation_id)
        except GoalError as exc:
            self._on_debug(f"tool_gateway: could not advance goal on observation completion: {exc}")
