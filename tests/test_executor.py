"""ActionExecutor: image-point mapping, action dispatch, invalid input safety."""

from dataclasses import dataclass

import pytest

from src.body.executor import ActionExecutor, ExecutorHooks, image_point_to_pan_tilt
from src.protocol.models import Action


@dataclass
class FakeLimits:
    lower: float
    upper: float
    velocity: float
    effort: float = 10.0


class FakeSim:
    def __init__(self):
        joints = ["base_yaw_joint", "shoulder_pitch_joint", "elbow_pitch_joint", "neck_yaw_joint", "head_pitch_joint"]
        self._limits = {name: FakeLimits(lower=-2.45, upper=2.45, velocity=3.0) for name in joints}
        self._angles = dict.fromkeys(joints, 0.0)
        self.light_calls = []

    def limits_for(self, name):
        return self._limits[name]

    def get_joint_angle(self, name):
        return self._angles[name]

    def get_all_joint_angles(self):
        return dict(self._angles)

    def set_joint_angle(self, name, angle):
        limits = self._limits[name]
        self._angles[name] = max(limits.lower, min(limits.upper, angle))

    def set_light(self, on, color=(1.0, 1.0, 1.0), brightness=1.0):
        self.light_calls.append((on, color, brightness))

    def forward(self):
        pass


def make_executor():
    sim = FakeSim()
    return ActionExecutor(sim, hooks=ExecutorHooks()), sim


def run_to_completion(executor: ActionExecutor, action: Action) -> None:
    executor.run(action)


# -- image_point_to_pan_tilt -------------------------------------------------------

def test_image_point_center_maps_to_zero_pan_tilt():
    pan, tilt = image_point_to_pan_tilt(0.5, 0.5)
    assert pan == pytest.approx(0.0)
    assert tilt == pytest.approx(0.0)


def test_image_point_left_edge_gives_positive_pan():
    """pan is documented as +left (see Action's look_at docstring) --
    an object at the left edge of the frame (x=0) should pan positive."""
    pan, _ = image_point_to_pan_tilt(0.0, 0.5)
    assert pan > 0


def test_image_point_right_edge_gives_negative_pan():
    pan, _ = image_point_to_pan_tilt(1.0, 0.5)
    assert pan < 0


def test_image_point_top_gives_positive_tilt_bottom_gives_negative():
    _, tilt_top = image_point_to_pan_tilt(0.5, 0.0)
    _, tilt_bottom = image_point_to_pan_tilt(0.5, 1.0)
    assert tilt_top > 0
    assert tilt_bottom < 0


def test_image_point_gains_are_configurable():
    pan_default, _ = image_point_to_pan_tilt(0.0, 0.5)
    pan_custom, _ = image_point_to_pan_tilt(0.0, 0.5, horizontal_gain=0.1)
    assert pan_custom < pan_default


# -- action dispatch safety --------------------------------------------------------

def test_look_at_clamps_pan_to_joint_limits():
    executor, sim = make_executor()
    run_to_completion(executor, Action(kind="look_at", params={"pan": 999.0, "tilt": 0.0}))
    assert sim.get_joint_angle("base_yaw_joint") == pytest.approx(2.45)


def test_unhandled_action_kind_is_caught_as_telemetry_not_a_crash():
    executor, sim = make_executor()
    bad_action = Action.model_construct(kind="not_a_real_kind", params={})  # bypass Action's own validation
    executor.run(bad_action)
    telemetry = executor.drain_telemetry()
    assert any(t.kind == "action_failed" for t in telemetry)


def test_set_light_dispatches_to_sim():
    executor, sim = make_executor()
    executor.run(Action(kind="set_light", params={"on": True, "color": [1.0, 0.5, 0.0], "brightness": 0.8}))
    assert sim.light_calls == [(True, (1.0, 0.5, 0.0), 0.8)]


def test_gesture_actions_all_complete_without_error():
    executor, sim = make_executor()
    for kind in ("nod", "shake_head", "excited", "curious", "think", "home", "idle_sway"):
        executor.run(Action(kind=kind, params={}))
        telemetry = executor.drain_telemetry()
        assert not any(t.kind == "action_failed" for t in telemetry), f"{kind} failed: {telemetry}"
