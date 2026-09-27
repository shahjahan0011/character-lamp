"""SceneObserver: ambient scene path + tool-driven request_observation path."""

import time
from unittest.mock import patch

import numpy as np

from src.character.memory import SceneMemory
from src.character.observation_registry import ObservationRegistry
from src.character.scene_observer import SceneObserver
from src.protocol.observation import DetectedObject, Observation
from src.speech.gemini_client import GeminiStreamError


def wait_until_not_busy(observer, timeout_s=2.0):
    # A short settle delay first: is_busy() can read False before the
    # worker thread has even picked up the just-submitted job, which
    # would make this return instantly (before real processing happens)
    # rather than actually waiting for it.
    time.sleep(0.05)
    deadline = time.monotonic() + timeout_s
    while observer.is_busy() and time.monotonic() < deadline:
        time.sleep(0.01)


def test_ambient_scene_observation_updates_memory_silently():
    memory = SceneMemory(persist_path=None)
    registry = ObservationRegistry()
    observer = SceneObserver(memory=memory, registry=registry, on_debug=print)
    fake_obs = Observation(
        purpose="scene", width=640, height=480,
        objects=[DetectedObject(label="Mug", attributes="red", image_x=0.5, image_y=0.5, confidence=0.9)],
    )
    with patch("src.character.scene_observer.vision.describe_scene", return_value=fake_obs):
        observer.submit_observation(np.zeros((4, 4, 3), dtype=np.uint8), purpose="scene")
        wait_until_not_busy(observer)

    result = observer.pop_new_observation()
    assert result is not None
    assert result.tool_call_id is None
    assert result.changed is True
    assert len(memory) == 1
    assert registry.get(fake_obs.observation_id) is not None
    observer.stop()


def test_tool_driven_observation_does_not_touch_memory_but_registers():
    memory = SceneMemory(persist_path=None)
    registry = ObservationRegistry()
    observer = SceneObserver(memory=memory, registry=registry, on_debug=print)
    fake_obs = Observation(
        purpose="goal_planning", width=640, height=480,
        objects=[DetectedObject(label="Mug", attributes="red", image_x=0.5, image_y=0.5, confidence=0.9)],
    )
    with patch("src.character.scene_observer.vision.describe_scene", return_value=fake_obs):
        observer.submit_observation(
            np.zeros((4, 4, 3), dtype=np.uint8), purpose="goal_planning", tool_call_id="call1", tool_name="request_observation"
        )
        wait_until_not_busy(observer)

    result = observer.pop_new_observation()
    assert result.tool_call_id == "call1"
    assert result.tool_name == "request_observation"
    assert len(memory) == 0  # goal_planning observations don't auto-populate memory
    assert registry.get(fake_obs.observation_id) is not None
    observer.stop()


def test_busy_observer_refuses_new_submission():
    observer = SceneObserver()
    with patch("src.character.scene_observer.vision.describe_scene", side_effect=lambda *a, **k: time.sleep(0.2) or Observation(purpose="scene", width=1, height=1)):
        assert observer.submit_observation(np.zeros((2, 2, 3), dtype=np.uint8)) is True
        time.sleep(0.02)
        assert observer.is_busy() is True
        assert observer.submit_observation(np.zeros((2, 2, 3), dtype=np.uint8)) is False
        wait_until_not_busy(observer)
    observer.stop()


def test_rate_limited_failure_is_reported_and_tagged_for_tool_call():
    observer = SceneObserver()

    def raise_rate_limit(*args, **kwargs):
        raise GeminiStreamError("rate_limit_exceeded", "slow down")

    with patch("src.character.scene_observer.vision.describe_scene", side_effect=raise_rate_limit):
        observer.submit_observation(
            np.zeros((2, 2, 3), dtype=np.uint8), purpose="goal_planning", tool_call_id="call2", tool_name="request_observation"
        )
        wait_until_not_busy(observer)

    assert observer.pop_failure() is True
    assert observer.pop_rate_limited() is True
    failed_call = observer.pop_failed_tool_call()
    assert failed_call is not None
    call_id, tool_name, message = failed_call
    assert call_id == "call2"
    assert tool_name == "request_observation"
    observer.stop()


def test_non_tool_failure_does_not_report_a_failed_tool_call():
    observer = SceneObserver()
    with patch("src.character.scene_observer.vision.describe_scene", side_effect=RuntimeError("boom")):
        observer.submit_observation(np.zeros((2, 2, 3), dtype=np.uint8), purpose="scene")
        wait_until_not_busy(observer)
    assert observer.pop_failure() is True
    assert observer.pop_failed_tool_call() is None
    observer.stop()
