"""Velocity-limited joint interpolation.

A real lamp arm can't teleport to a new pose. Every motion we command goes
through here so it always respects each joint's URDF velocity limit -- this
is what "physical reasoning" means for a kinematically-driven demo: we can't
simulate real torque/dynamics, but we can and do refuse to move a joint
faster than the real actuator could.
"""

from __future__ import annotations

from dataclasses import dataclass

from .sim import CONTROLLED_JOINTS, LampSimulator

CONTROL_HZ = 50.0
DT = 1.0 / CONTROL_HZ


@dataclass
class JointMove:
    start: float
    target: float
    duration: float  # seconds, derived from the joint's velocity limit
    elapsed: float = 0.0

    def done(self) -> bool:
        return self.elapsed >= self.duration

    def angle_at(self, t: float) -> float:
        if self.duration <= 0.0:
            return self.target
        frac = min(1.0, t / self.duration)
        # Smoothstep, not linear: zero velocity at both ends of the move, so
        # consecutive queued moves don't jerk at the seams.
        smooth = frac * frac * (3 - 2 * frac)
        return self.start + (self.target - self.start) * smooth


class TrajectoryPlayer:
    """Drives every controlled joint of a LampSimulator toward a target pose,
    each at its own velocity-limited pace, and reports when the whole move
    is done."""

    def __init__(self, sim: LampSimulator):
        self.sim = sim
        self._moves: dict[str, JointMove] = {}

    def move_to(self, targets: dict[str, float], speed_scale: float = 1.0) -> None:
        """Start a new move for the given joints (others hold their current
        position). speed_scale < 1.0 slows the whole move down (e.g. for a
        deliberate, gentle gesture); it never exceeds 1.0, since that would
        violate the joint's velocity limit.

        All joints in this call share one duration -- the slowest joint's own
        velocity-limited time. Without this, a multi-joint move like "return
        home" has each joint arrive at a different moment (whichever has the
        shortest distance-over-velocity finishes first), which reads as
        uncoordinated/jerky rather than one settling motion. Sharing the
        duration only ever slows other joints down to match, never speeds
        any joint past its own limit."""
        speed_scale = min(1.0, max(0.01, speed_scale))
        pending: dict[str, tuple[float, float, float]] = {}  # name -> (start, target, natural_duration)
        for name, target in targets.items():
            limits = self.sim.limits_for(name)
            clamped_target = max(limits.lower, min(limits.upper, target))
            start = self.sim.get_joint_angle(name)
            distance = abs(clamped_target - start)
            max_velocity = limits.velocity * speed_scale
            natural_duration = distance / max_velocity if max_velocity > 0 else 0.0
            pending[name] = (start, clamped_target, natural_duration)

        group_duration = max((d for _, _, d in pending.values()), default=0.0)
        for name, (start, clamped_target, _) in pending.items():
            self._moves[name] = JointMove(start=start, target=clamped_target, duration=group_duration)

    def is_moving(self) -> bool:
        return any(not m.done() for m in self._moves.values())

    def step(self, dt: float = DT) -> None:
        """Advance all in-flight moves by dt and push the new angles into the
        simulator. Call sim.forward() afterward to update rendered poses."""
        for name, move in list(self._moves.items()):
            move.elapsed = min(move.duration, move.elapsed + dt)
            self.sim.set_joint_angle(name, move.angle_at(move.elapsed))

    def home(self) -> None:
        self.move_to({name: 0.0 for name in CONTROLLED_JOINTS})
