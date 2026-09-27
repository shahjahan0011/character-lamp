"""Trajectory velocity/limit safety -- no PyBullet needed, this is pure math."""

from dataclasses import dataclass

import pytest

from src.body.trajectory import DT, SMOOTHSTEP_PEAK_FACTOR, TrajectoryPlayer


@dataclass
class FakeLimits:
    lower: float
    upper: float
    velocity: float
    effort: float = 10.0


class FakeSim:
    """Implements just the TrajectoryPlayer-facing interface -- no PyBullet."""

    def __init__(self, limits: dict[str, FakeLimits]):
        self._limits = limits
        self._angles = dict.fromkeys(limits, 0.0)

    def limits_for(self, name):
        return self._limits[name]

    def get_joint_angle(self, name):
        return self._angles[name]

    def set_joint_angle(self, name, angle):
        limits = self._limits[name]
        self._angles[name] = max(limits.lower, min(limits.upper, angle))

    def forward(self):
        pass


def make_player(velocity=2.0, lower=-2.0, upper=2.0, joints=("j1",)):
    sim = FakeSim({name: FakeLimits(lower=lower, upper=upper, velocity=velocity) for name in joints})
    return TrajectoryPlayer(sim), sim


def test_joint_target_clamps_to_urdf_limits():
    player, sim = make_player(lower=-1.0, upper=1.0)
    player.move_to({"j1": 5.0})  # way past upper limit
    while player.is_moving():
        player.step(DT)
    assert sim.get_joint_angle("j1") == 1.0


def test_smoothstep_peak_velocity_never_exceeds_configured_limit():
    max_velocity = 2.0
    player, sim = make_player(velocity=max_velocity, lower=-10.0, upper=10.0)
    player.move_to({"j1": 5.0})
    move = player._moves["j1"]

    # Sample velocity finely across the whole move via the analytic
    # smoothstep derivative (angle_at's own formula), not just a handful
    # of simulation steps -- this is what actually proves the peak (at
    # t=duration/2) stays within the limit, not just the endpoints.
    fine_dt = move.duration / 2000
    prev_angle = move.angle_at(0.0)
    peak_velocity = 0.0
    t = fine_dt
    while t <= move.duration:
        angle = move.angle_at(t)
        velocity = abs(angle - prev_angle) / fine_dt
        peak_velocity = max(peak_velocity, velocity)
        prev_angle = angle
        t += fine_dt

    # Small numerical-differentiation tolerance, not a loosened safety margin.
    assert peak_velocity <= max_velocity * 1.01, f"peak velocity {peak_velocity} exceeded limit {max_velocity}"


def test_duration_includes_smoothstep_peak_factor():
    max_velocity = 2.0
    player, sim = make_player(velocity=max_velocity)
    player.move_to({"j1": 1.0})
    move = player._moves["j1"]
    expected_duration = SMOOTHSTEP_PEAK_FACTOR * 1.0 / max_velocity
    assert move.duration == pytest.approx(expected_duration)


def test_multi_joint_move_shares_one_synchronized_duration():
    player, sim = make_player(velocity=2.0, joints=("fast", "slow"))
    # "fast" has a short distance (quick on its own); "slow" has a long
    # distance (needs more time) -- both must finish at the same instant.
    player.move_to({"fast": 0.2, "slow": 3.0})
    assert player._moves["fast"].duration == player._moves["slow"].duration
    assert player._moves["slow"].duration > 0


def test_zero_velocity_limit_does_not_crash():
    player, sim = make_player(velocity=0.0)
    player.move_to({"j1": 1.0})
    assert player._moves["j1"].duration == 0.0
    assert player._moves["j1"].done() is True


def test_speed_scale_never_exceeds_one():
    player, sim = make_player(velocity=2.0)
    player.move_to({"j1": 1.0}, speed_scale=5.0)  # attempted > 1.0
    fast_duration = player._moves["j1"].duration
    player.move_to({"j1": 1.0}, speed_scale=1.0)
    normal_duration = player._moves["j1"].duration
    assert fast_duration == pytest.approx(normal_duration)  # clamped, not actually faster
