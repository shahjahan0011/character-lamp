"""GoalCoordinator: the actual enforcement of demo moment 5's invariants."""

import pytest

from src.character.goal_coordinator import GoalCoordinator, GoalError
from src.character.observation_registry import ObservationRegistry
from src.protocol.observation import Observation


def make_obs(purpose, captured_at_monotonic):
    return Observation(purpose=purpose, width=640, height=480, objects=[], captured_at_monotonic=captured_at_monotonic)


def test_happy_path_reaches_verified(monkeypatch):
    registry = ObservationRegistry()
    coordinator = GoalCoordinator(registry)
    goal = coordinator.start_goal()

    monkeypatch.setattr("time.monotonic", lambda: 100.0)
    plan_obs = make_obs("goal_planning", 100.0)
    registry.add(plan_obs)
    coordinator.record_planning_observation(goal.goal_id, plan_obs.observation_id)

    monkeypatch.setattr("time.monotonic", lambda: 101.0)
    coordinator.record_action_completed(goal.goal_id)
    assert coordinator.get(goal.goal_id).last_action_completed_at == 101.0

    monkeypatch.setattr("time.monotonic", lambda: 102.0)
    verify_obs = make_obs("goal_verification", 102.0)
    registry.add(verify_obs)
    coordinator.record_verification_observation(goal.goal_id, verify_obs.observation_id)

    final = coordinator.finish(
        goal.goal_id, verify_obs.observation_id, success=True, evidence="mug is visible", confidence=0.9
    )
    assert final.stage == "verified"


def test_rejects_action_without_planning_observation():
    registry = ObservationRegistry()
    coordinator = GoalCoordinator(registry)
    goal = coordinator.start_goal()
    with pytest.raises(GoalError, match="planning"):
        coordinator.record_action_completed(goal.goal_id)


def test_rejects_planning_with_unknown_observation():
    registry = ObservationRegistry()
    coordinator = GoalCoordinator(registry)
    goal = coordinator.start_goal()
    with pytest.raises(GoalError, match="unknown or stale"):
        coordinator.record_planning_observation(goal.goal_id, "nonexistent-observation-id")


def test_rejects_stale_planning_observation(monkeypatch):
    registry = ObservationRegistry(max_age_s=10.0)
    coordinator = GoalCoordinator(registry)
    goal = coordinator.start_goal()
    monkeypatch.setattr("time.monotonic", lambda: 0.0)
    obs = make_obs("goal_planning", 0.0)
    registry.add(obs)
    monkeypatch.setattr("time.monotonic", lambda: 100.0)  # long past max_age_s
    with pytest.raises(GoalError, match="unknown or stale"):
        coordinator.record_planning_observation(goal.goal_id, obs.observation_id)


def test_rejects_planning_observation_with_wrong_purpose():
    registry = ObservationRegistry()
    coordinator = GoalCoordinator(registry)
    goal = coordinator.start_goal()
    import time as _time

    wrong_purpose_obs = make_obs("scene", _time.monotonic())
    registry.add(wrong_purpose_obs)
    with pytest.raises(GoalError, match="goal_planning"):
        coordinator.record_planning_observation(goal.goal_id, wrong_purpose_obs.observation_id)


def test_rejects_verification_before_any_action():
    registry = ObservationRegistry()
    coordinator = GoalCoordinator(registry)
    goal = coordinator.start_goal()
    import time as _time

    now = _time.monotonic()
    plan_obs = make_obs("goal_planning", now)
    registry.add(plan_obs)
    coordinator.record_planning_observation(goal.goal_id, plan_obs.observation_id)

    verify_obs = make_obs("goal_verification", now + 1.0)
    registry.add(verify_obs)
    with pytest.raises(GoalError, match="at least one validated action"):
        coordinator.record_verification_observation(goal.goal_id, verify_obs.observation_id)


def test_rejects_verification_observation_not_newer_than_final_action(monkeypatch):
    """The verification observation must be captured strictly *after* the
    final action completed -- an observation from before (or at) that
    moment can't prove the action actually happened."""
    registry = ObservationRegistry()
    coordinator = GoalCoordinator(registry)
    goal = coordinator.start_goal()

    monkeypatch.setattr("time.monotonic", lambda: 10.0)
    plan_obs = make_obs("goal_planning", 10.0)
    registry.add(plan_obs)
    coordinator.record_planning_observation(goal.goal_id, plan_obs.observation_id)

    monkeypatch.setattr("time.monotonic", lambda: 20.0)
    coordinator.record_action_completed(goal.goal_id)  # last_action_completed_at = 20.0

    # Verification observation captured BEFORE the action completed.
    stale_verify_obs = make_obs("goal_verification", 15.0)
    registry.add(stale_verify_obs)
    with pytest.raises(GoalError, match="after the final action"):
        coordinator.record_verification_observation(goal.goal_id, stale_verify_obs.observation_id)


def test_rejects_finish_goal_without_verification_observation():
    registry = ObservationRegistry()
    coordinator = GoalCoordinator(registry)
    goal = coordinator.start_goal()
    with pytest.raises(GoalError, match="goal_verification"):
        coordinator.finish(goal.goal_id, "some-observation-id", success=True, evidence="x", confidence=0.9)


def test_rejects_finish_goal_with_mismatched_observation_id(monkeypatch):
    registry = ObservationRegistry()
    coordinator = GoalCoordinator(registry)
    goal = coordinator.start_goal()

    monkeypatch.setattr("time.monotonic", lambda: 0.0)
    plan_obs = make_obs("goal_planning", 0.0)
    registry.add(plan_obs)
    coordinator.record_planning_observation(goal.goal_id, plan_obs.observation_id)
    monkeypatch.setattr("time.monotonic", lambda: 1.0)
    coordinator.record_action_completed(goal.goal_id)
    monkeypatch.setattr("time.monotonic", lambda: 2.0)
    verify_obs = make_obs("goal_verification", 2.0)
    registry.add(verify_obs)
    coordinator.record_verification_observation(goal.goal_id, verify_obs.observation_id)

    with pytest.raises(GoalError, match="must match"):
        coordinator.finish(goal.goal_id, "a-different-observation-id", success=True, evidence="x", confidence=0.9)


def test_one_inconclusive_retry_before_graceful_failure(monkeypatch):
    registry = ObservationRegistry()
    coordinator = GoalCoordinator(registry, max_retries=1)
    goal = coordinator.start_goal()

    def do_full_cycle(t):
        monkeypatch.setattr("time.monotonic", lambda: t)
        plan_obs = make_obs("goal_planning", t)
        registry.add(plan_obs)
        coordinator.record_planning_observation(goal.goal_id, plan_obs.observation_id)
        monkeypatch.setattr("time.monotonic", lambda: t + 1)
        coordinator.record_action_completed(goal.goal_id)
        monkeypatch.setattr("time.monotonic", lambda: t + 2)
        verify_obs = make_obs("goal_verification", t + 2)
        registry.add(verify_obs)
        coordinator.record_verification_observation(goal.goal_id, verify_obs.observation_id)
        return verify_obs

    verify_obs_1 = do_full_cycle(100.0)
    result_1 = coordinator.finish(goal.goal_id, verify_obs_1.observation_id, success=False, evidence="not found", confidence=0.3)
    assert result_1.stage == "planning"  # retried, not failed yet
    assert result_1.retries_used == 1

    verify_obs_2 = do_full_cycle(200.0)
    result_2 = coordinator.finish(goal.goal_id, verify_obs_2.observation_id, success=False, evidence="still not found", confidence=0.2)
    assert result_2.stage == "failed"  # retries exhausted


def test_unknown_goal_id_is_rejected():
    registry = ObservationRegistry()
    coordinator = GoalCoordinator(registry)
    with pytest.raises(GoalError, match="unknown goal_id"):
        coordinator.record_action_completed("never-started")
