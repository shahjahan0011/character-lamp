"""ToolGateway: validated execution + goal-workflow wiring."""

from unittest.mock import MagicMock

import numpy as np

from src.character.goal_coordinator import GoalCoordinator
from src.character.memory import SceneMemory
from src.character.observation_registry import ObservationRegistry
from src.character.tool_gateway import ToolGateway
from src.protocol.observation import DetectedObject, Observation


def make_gateway():
    executor = MagicMock()
    observer = MagicMock()
    registry = ObservationRegistry()
    memory = SceneMemory(persist_path=None)
    goals = GoalCoordinator(registry)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    gateway = ToolGateway(executor, observer, registry, memory, goals, get_latest_frame=lambda: frame, on_debug=print)
    return gateway, executor, observer, registry, memory, goals


def add_observation(registry, purpose="scene", objects=None):
    obs = Observation(purpose=purpose, width=640, height=480, objects=objects or [])
    registry.add(obs)
    return obs


def test_unknown_tool_name_returns_structured_error():
    gateway, *_ = make_gateway()
    result = gateway.execute("delete_all_files", {}, call_id="c1")
    assert result == {"ok": False, "error": "unknown tool 'delete_all_files'"}


def test_invalid_arguments_return_structured_error_not_exception():
    gateway, *_ = make_gateway()
    result = gateway.execute("perform_gesture", {"name": "backflip"}, call_id="c1")
    assert result["ok"] is False
    assert "invalid arguments" in result["error"]


def test_recall_memory_returns_structured_records():
    gateway, _, _, _, memory, _ = make_gateway()
    memory.remember([DetectedObject(label="Red mug", attributes="ceramic", image_x=0.5, image_y=0.5, confidence=0.9)], source_observation_id="obs1")
    result = gateway.execute("recall_memory", {"query": "mug", "limit": 5}, call_id="c1")
    assert result["ok"] is True
    assert len(result["records"]) == 1
    assert result["records"][0]["label"] == "Red mug"


def test_remember_object_rejects_unknown_observation():
    gateway, *_ = make_gateway()
    result = gateway.execute(
        "remember_object",
        {"observation_id": "nope", "label": "mug", "color": None, "attributes": "", "image_x": 0.5, "image_y": 0.5, "confidence": 0.9, "notes": None},
        call_id="c1",
    )
    assert result["ok"] is False
    assert "unknown or stale" in result["error"]


def test_remember_object_succeeds_with_fresh_observation():
    gateway, _, _, registry, memory, _ = make_gateway()
    obs = add_observation(registry, purpose="object_memory")
    result = gateway.execute(
        "remember_object",
        {"observation_id": obs.observation_id, "label": "mug", "color": "red", "attributes": "ceramic", "image_x": 0.5, "image_y": 0.5, "confidence": 0.9, "notes": None},
        call_id="c1",
    )
    assert result == {"ok": True, "label": "mug"}
    assert len(memory) == 1


def test_look_at_image_point_rejects_stale_observation():
    gateway, executor, *_ = make_gateway()
    result = gateway.execute("look_at_image_point", {"observation_id": "unknown", "x": 0.5, "y": 0.5}, call_id="c1")
    assert result["ok"] is False
    executor.run.assert_not_called()


def test_look_at_image_point_dispatches_point_at_action():
    gateway, executor, _, registry, *_ = make_gateway()
    obs = add_observation(registry, purpose="goal_planning")
    result = gateway.execute("look_at_image_point", {"observation_id": obs.observation_id, "x": 0.0, "y": 0.5}, call_id="c1")
    assert result["ok"] is True
    executor.run.assert_called_once()
    action = executor.run.call_args.args[0]
    assert action.kind == "point_at"
    assert action.params["pan"] > 0  # left edge -> positive pan, per executor.py's convention


def test_set_light_dispatches_set_light_action():
    gateway, executor, *_ = make_gateway()
    result = gateway.execute("set_light", {"brightness": 0.7, "mode": "steady", "duration_ms": 200}, call_id="c1")
    assert result == {"ok": True}
    action = executor.run.call_args.args[0]
    assert action.kind == "set_light"
    assert action.params["brightness"] == 0.7


def test_perform_gesture_dispatches_gesture_action():
    gateway, executor, *_ = make_gateway()
    result = gateway.execute("perform_gesture", {"name": "nod", "intensity": 1.0}, call_id="c1")
    assert result == {"ok": True}
    action = executor.run.call_args.args[0]
    assert action.kind == "nod"


def test_request_observation_queues_job_and_returns_none():
    gateway, _, observer, *_ = make_gateway()
    observer.submit_observation.return_value = True
    result = gateway.execute("request_observation", {"purpose": "scene"}, call_id="c1")
    assert result is None
    observer.submit_observation.assert_called_once()
    _, kwargs = observer.submit_observation.call_args
    assert kwargs["purpose"] == "scene"
    assert kwargs["tool_call_id"] == "c1"


def test_request_observation_reports_error_when_observer_busy():
    gateway, _, observer, *_ = make_gateway()
    observer.submit_observation.return_value = False
    result = gateway.execute("request_observation", {"purpose": "scene"}, call_id="c1")
    assert result["ok"] is False


def test_request_observation_reports_error_with_no_camera_frame():
    executor = MagicMock()
    observer = MagicMock()
    registry = ObservationRegistry()
    memory = SceneMemory(persist_path=None)
    goals = GoalCoordinator(registry)
    gateway = ToolGateway(executor, observer, registry, memory, goals, get_latest_frame=lambda: None)
    result = gateway.execute("request_observation", {"purpose": "scene"}, call_id="c1")
    assert result["ok"] is False
    assert "no camera frame" in result["error"]


# -- full goal-workflow happy path via the gateway ----------------------------------

def test_goal_workflow_happy_path_through_gateway():
    gateway, executor, observer, registry, memory, goals = make_gateway()

    # 1. goal_planning observation request starts a new goal.
    observer.submit_observation.return_value = True
    result = gateway.execute("request_observation", {"purpose": "goal_planning"}, call_id="c1")
    assert result is None
    goal_id = gateway._current_goal_id
    assert goal_id is not None
    assert goals.get(goal_id).stage == "observing"

    # 2. the observation "completes" -- fsm.py would call this.
    plan_obs = add_observation(registry, purpose="goal_planning", objects=[
        DetectedObject(label="Red mug", attributes="ceramic", image_x=0.3, image_y=0.4, confidence=0.9)
    ])
    gateway.complete_observation_for_goal(plan_obs.observation_id, "goal_planning")
    assert goals.get(goal_id).stage == "planning"

    # 3. the model points at the mug using that observation's coordinates.
    result = gateway.execute(
        "look_at_image_point", {"observation_id": plan_obs.observation_id, "x": 0.3, "y": 0.4}, call_id="c2"
    )
    assert result["ok"] is True
    assert goals.get(goal_id).stage == "acting"
    assert goals.get(goal_id).actions_completed == 1

    # 4. request a goal_verification observation.
    result = gateway.execute("request_observation", {"purpose": "goal_verification"}, call_id="c3")
    assert result is None
    verify_obs = add_observation(registry, purpose="goal_verification", objects=[
        DetectedObject(label="Red mug", attributes="ceramic", image_x=0.3, image_y=0.4, confidence=0.9)
    ])
    gateway.complete_observation_for_goal(verify_obs.observation_id, "goal_verification")
    assert goals.get(goal_id).stage == "reobserving"

    # 5. finish_goal succeeds only with the real verification observation.
    result = gateway.execute(
        "finish_goal",
        {"goal_id": goal_id, "post_action_observation_id": verify_obs.observation_id, "success": True, "evidence": "mug visible", "confidence": 0.9},
        call_id="c4",
    )
    assert result["ok"] is True
    assert result["stage"] == "verified"
    assert gateway._current_goal_id is None  # cleared once resolved


def test_finish_goal_before_workflow_complete_is_rejected():
    gateway, *_ = make_gateway()
    result = gateway.execute(
        "finish_goal",
        {"goal_id": "never-started", "post_action_observation_id": "x", "success": True, "evidence": "e", "confidence": 0.9},
        call_id="c1",
    )
    assert result["ok"] is False
