"""ObservationRegistry: freshness, unknown-id rejection, ordering."""

from src.character.observation_registry import ObservationRegistry
from src.protocol.observation import Observation


def make_obs(purpose="scene", captured_at_monotonic=0.0):
    return Observation(
        purpose=purpose, width=640, height=480, objects=[], captured_at_monotonic=captured_at_monotonic
    )


def test_unknown_observation_id_is_not_fresh():
    registry = ObservationRegistry()
    assert registry.is_fresh("does-not-exist") is False
    assert registry.get("does-not-exist") is None


def test_fresh_observation_within_max_age(monkeypatch):
    registry = ObservationRegistry(max_age_s=30.0)
    obs = make_obs(captured_at_monotonic=100.0)
    registry.add(obs)
    monkeypatch.setattr("time.monotonic", lambda: 110.0)
    assert registry.is_fresh(obs.observation_id) is True


def test_stale_observation_beyond_max_age(monkeypatch):
    registry = ObservationRegistry(max_age_s=30.0)
    obs = make_obs(captured_at_monotonic=100.0)
    registry.add(obs)
    monkeypatch.setattr("time.monotonic", lambda: 200.0)
    assert registry.is_fresh(obs.observation_id) is False


def test_latest_filters_by_purpose():
    registry = ObservationRegistry()
    scene_obs = make_obs(purpose="scene", captured_at_monotonic=1.0)
    plan_obs = make_obs(purpose="goal_planning", captured_at_monotonic=2.0)
    registry.add(scene_obs)
    registry.add(plan_obs)
    assert registry.latest(purpose="goal_planning").observation_id == plan_obs.observation_id
    assert registry.latest(purpose="scene").observation_id == scene_obs.observation_id
    assert registry.latest().observation_id == plan_obs.observation_id  # most recently added


def test_max_entries_evicts_oldest():
    registry = ObservationRegistry(max_entries=2)
    first = make_obs(captured_at_monotonic=1.0)
    second = make_obs(captured_at_monotonic=2.0)
    third = make_obs(captured_at_monotonic=3.0)
    registry.add(first)
    registry.add(second)
    registry.add(third)
    assert registry.get(first.observation_id) is None
    assert registry.get(second.observation_id) is not None
    assert registry.get(third.observation_id) is not None


def test_is_newer_than():
    registry = ObservationRegistry()
    obs = make_obs(captured_at_monotonic=100.0)
    registry.add(obs)
    assert registry.is_newer_than(obs.observation_id, than_monotonic=99.0) is True
    assert registry.is_newer_than(obs.observation_id, than_monotonic=100.0) is False
    assert registry.is_newer_than(obs.observation_id, than_monotonic=101.0) is False
    assert registry.is_newer_than("unknown-id", than_monotonic=0.0) is False
