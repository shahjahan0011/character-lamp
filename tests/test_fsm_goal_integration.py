"""fsm.py's integration with ToolGateway/GoalCoordinator/ObservationRegistry
-- the actual runtime wiring for demo moment 5, all mocked (no camera,
mic, speaker, display, network, or API key)."""

import time
from unittest.mock import MagicMock

import numpy as np

from src.character.character_state import CharacterState
from src.character.fsm import CharacterOrchestrator
from src.character.memory import SceneMemory
from src.character.scene_observer import ObservationResult
from src.perception.engagement import EngagementState
from src.protocol.observation import DetectedObject, Observation
from src.speech.live_client import LiveEvent


def make_orchestrator(engaged=True):
    executor = MagicMock()
    executor.sim.is_connected.return_value = True
    executor.drain_telemetry.return_value = []
    watcher = MagicMock()
    watcher.get_latest_frame.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    watcher.get_state.return_value = EngagementState(engaged=engaged, face_x_frac=0.0, changed_at=time.time())
    live_client = MagicMock()
    live_client.poll_events.return_value = []
    mixer = MagicMock()
    mixer.output_pending = False
    mixer.pop_xrun_counts.return_value = (0, 0)
    memory = SceneMemory(persist_path=None)
    observer = MagicMock()
    observer.submit_observation.return_value = True
    observer.pop_failure.return_value = False
    observer.pop_new_observation.return_value = None
    orch = CharacterOrchestrator(
        executor, watcher, live_client=live_client, audio_mixer=mixer, observer=observer,
        memory=memory, on_debug=print
    )
    return orch, executor, watcher, live_client, mixer, memory


def test_non_blocking_gesture_is_acknowledged_before_motion_runs():
    orch, executor, watcher, live_client, mixer, memory = make_orchestrator()
    orch.tick()  # ENGAGE
    order = []
    live_client.submit_tool_result.side_effect = lambda *args: order.append("tool_result")
    executor.run.side_effect = lambda action, **kwargs: order.append(action.kind)
    live_client.poll_events.return_value = [
        LiveEvent(kind="tool_call", tool_name="perform_gesture", tool_args={"name": "nod"}, tool_call_id="c1")
    ]
    orch.tick()
    live_client.submit_tool_result.assert_called_with("c1", "perform_gesture", {"ok": True, "status": "started"})
    assert order.index("tool_result") < order.index("nod")
    gesture_calls = [c for c in executor.run.call_args_list if getattr(c.args[0], "kind", None) == "nod"]
    assert len(gesture_calls) >= 1


def test_tool_call_with_invalid_gesture_is_rejected_via_gateway():
    orch, executor, watcher, live_client, mixer, memory = make_orchestrator()
    orch.tick()
    live_client.poll_events.return_value = [
        LiveEvent(kind="tool_call", tool_name="perform_gesture", tool_args={"name": "backflip"}, tool_call_id="c1")
    ]
    orch.tick()
    call_id, name, result = live_client.submit_tool_result.call_args.args
    assert result["ok"] is False


def test_request_observation_tool_call_queues_and_completes_asynchronously():
    orch, executor, watcher, live_client, mixer, memory = make_orchestrator()
    orch.tick()
    live_client.poll_events.return_value = [
        LiveEvent(kind="tool_call", tool_name="request_observation", tool_args={"purpose": "scene"}, tool_call_id="c1")
    ]
    orch.tick()
    # Should NOT have submitted a result yet -- it's queued on SceneObserver.
    assert live_client.submit_tool_result.call_count == 0

    # Simulate the observer finishing (bypass the real thread for determinism).
    fake_obs = Observation(purpose="scene", width=640, height=480, objects=[])
    orch.observer.pop_new_observation = MagicMock(
        return_value=ObservationResult(observation=fake_obs, changed=False, tool_call_id="c1", tool_name="request_observation")
    )
    orch.observer.pop_failure = MagicMock(return_value=False)
    live_client.poll_events.return_value = []
    orch.tick()
    live_client.submit_tool_result.assert_called_once()
    call_id, name, result = live_client.submit_tool_result.call_args.args
    assert call_id == "c1"
    assert result["ok"] is True
    assert result["observation_id"] == fake_obs.observation_id


def test_failed_tool_driven_observation_completes_with_structured_error():
    orch, executor, watcher, live_client, mixer, memory = make_orchestrator()
    orch.tick()
    orch.observer.pop_failure = MagicMock(return_value=True)
    orch.observer.pop_rate_limited = MagicMock(return_value=False)
    orch.observer.pop_failed_tool_call = MagicMock(return_value=("c1", "request_observation", "network error"))
    orch.observer.pop_new_observation = MagicMock(return_value=None)
    live_client.poll_events.return_value = []
    orch.tick()
    live_client.submit_tool_result.assert_called_with("c1", "request_observation", {"ok": False, "error": "network error"})


def test_full_goal_workflow_through_fsm_tick():
    """The same happy path verified against the real API earlier, now
    driven purely through CharacterOrchestrator.tick() with everything
    else mocked."""
    orch, executor, watcher, live_client, mixer, memory = make_orchestrator()
    orch.tick()  # ENGAGE

    # 1. request_observation(goal_planning) starts a goal.
    live_client.poll_events.return_value = [
        LiveEvent(kind="tool_call", tool_name="request_observation", tool_args={"purpose": "goal_planning"}, tool_call_id="c1")
    ]
    orch.tick()
    goal_id = orch.gateway._current_goal_id
    assert goal_id is not None

    # 2. observation completes (registering it is what the real
    # SceneObserver._process() does before publishing the result).
    plan_obs = Observation(
        purpose="goal_planning", width=640, height=480,
        objects=[DetectedObject(label="Red mug", attributes="", image_x=0.3, image_y=0.4, confidence=0.9)],
    )
    orch.registry.add(plan_obs)
    orch.observer.pop_new_observation = MagicMock(
        return_value=ObservationResult(observation=plan_obs, changed=False, tool_call_id="c1", tool_name="request_observation")
    )
    orch.observer.pop_failure = MagicMock(return_value=False)
    live_client.poll_events.return_value = []
    orch.tick()
    assert orch.goals.get(goal_id).stage == "planning"

    # 3. look_at_image_point using that observation.
    orch.observer.pop_new_observation = MagicMock(return_value=None)
    live_client.poll_events.return_value = [
        LiveEvent(kind="tool_call", tool_name="look_at_image_point",
                  tool_args={"observation_id": plan_obs.observation_id, "x": 0.3, "y": 0.4}, tool_call_id="c2")
    ]
    orch.tick()
    assert orch.goals.get(goal_id).stage == "acting"
    point_calls = [c for c in executor.run.call_args_list if getattr(c.args[0], "kind", None) == "point_at"]
    assert len(point_calls) >= 1

    # 4. request_observation(goal_verification).
    live_client.poll_events.return_value = [
        LiveEvent(kind="tool_call", tool_name="request_observation", tool_args={"purpose": "goal_verification"}, tool_call_id="c3")
    ]
    orch.tick()
    verify_obs = Observation(
        purpose="goal_verification", width=640, height=480,
        objects=[DetectedObject(label="Red mug", attributes="", image_x=0.3, image_y=0.4, confidence=0.9)],
    )
    orch.registry.add(verify_obs)
    orch.observer.pop_new_observation = MagicMock(
        return_value=ObservationResult(observation=verify_obs, changed=False, tool_call_id="c3", tool_name="request_observation")
    )
    live_client.poll_events.return_value = []
    orch.tick()
    assert orch.goals.get(goal_id).stage == "reobserving"

    # 5. finish_goal.
    orch.observer.pop_new_observation = MagicMock(return_value=None)
    live_client.poll_events.return_value = [
        LiveEvent(kind="tool_call", tool_name="finish_goal",
                  tool_args={"goal_id": goal_id, "post_action_observation_id": verify_obs.observation_id,
                             "success": True, "evidence": "visible", "confidence": 0.9},
                  tool_call_id="c4")
    ]
    orch.tick()
    assert orch.goals.get(goal_id).stage == "verified"
    call_id, name, result = live_client.submit_tool_result.call_args.args
    assert result["ok"] is True
    assert result["stage"] == "verified"


def test_disengage_clears_live_engagement_before_blocking_home_motion():
    """The actual bug this guards: audio permission must be revoked
    BEFORE the blocking home gesture runs, not after -- otherwise late
    audio could keep playing for that whole (blocking) duration."""
    orch, executor, watcher, live_client, mixer, memory = make_orchestrator(engaged=True)
    orch.tick()  # ENGAGE
    live_client.set_engaged.reset_mock()
    call_order = []
    live_client.set_engaged.side_effect = lambda v: call_order.append(("set_engaged", v))
    executor.run.side_effect = lambda action, **k: call_order.append(("run", action.kind))

    watcher.get_state.return_value = EngagementState(engaged=False, face_x_frac=0.0, changed_at=time.time())
    orch.tick()  # DISENGAGE

    home_index = next(i for i, (kind, val) in enumerate(call_order) if kind == "run" and val == "home")
    engaged_false_index = next(i for i, (kind, val) in enumerate(call_order) if kind == "set_engaged" and val is False)
    assert engaged_false_index < home_index, "set_engaged(False) must happen before the blocking home motion"


def test_state_machine_reaches_engaged_on_engage():
    orch, *_ = make_orchestrator(engaged=True)
    orch.tick()
    assert orch.state_machine.state == CharacterState.ENGAGED


def test_state_machine_reaches_dormant_on_disengage():
    orch, executor, watcher, live_client, mixer, memory = make_orchestrator(engaged=True)
    orch.tick()
    watcher.get_state.return_value = EngagementState(engaged=False, face_x_frac=0.0, changed_at=time.time())
    orch.tick()
    assert orch.state_machine.state == CharacterState.DORMANT
