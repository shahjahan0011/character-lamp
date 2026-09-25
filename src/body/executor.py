"""Turns Action messages into calls on the simulator.

This is the model/body boundary in code: everything above this file (the
character brain, any LLM/VLM) only ever produces `Action` objects from
src.protocol.models. This file is the only place that knows how a named
action maps to joint targets, light state, or a sound file -- the brain
never sees a joint name.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from src.protocol.models import Action, Telemetry

from .sim import LampSimulator
from .trajectory import DT, TrajectoryPlayer

# At all-zero joint angles the arm is fully extended straight up (each
# segment's local frame stacks with no bend), which does not read as "a lamp
# looking at you" no matter which way the base turns -- the head just spins
# around while still pointing at the ceiling. A real desk lamp leans its arm
# down toward what it's looking at, so "looking" bends the shoulder/elbow
# into a gentle forward pose that faces the shade opening at the viewer.
#
# History, because this took three tries and each one taught something:
#   1. shoulder=0.7/elbow=-1.2 looked like a reasonable lean in a single
#      screenshot, but the shade's opening was never actually checked --
#      it was pure side profile, never facing forward at all.
#   2. shoulder=-0.6/elbow=-1.7 (found by sweeping with the light on, so
#      the opening's direction was actually verified) bent the head down
#      to world Z=0.005 -- level with the floor, clipping through it.
#   3. shoulder=0.2/elbow=-0.9/neck_yaw=1.0 fixed both of those (checked
#      height AND facing this time), but neck_yaw is literally "swivel the
#      head sideways relative to the arm" -- it looked like glancing over
#      one's shoulder, not looking straight at someone. Wrong joint for
#      the job even though it scored well numerically.
#
# The actual fix: stop picking a camera angle first and then contorting the
# robot to face it. Instead, pick a natural, gentle pose with neck_yaw=0
# (no twist), then point the camera at wherever *that* pose actually faces
# -- found by rendering across the full 360-degree yaw range and scoring
# which angle sees the lit shade opening (see sim.py's render() default,
# yaw=270, which was derived the same way). This pose keeps the head above
# Z=0.6 and needs no per-frame neck compensation at all.
ATTENTIVE_SHOULDER_PITCH = 0.3
ATTENTIVE_ELBOW_PITCH = -0.6


# look_at / point_at pan+tilt (radians) -> joint targets. pan drives the base
# yaw (left/right), tilt fine-tunes head pitch (up/down) on top of the
# facing-the-viewer attentive pose above. neck_yaw stays at 0 -- it's not
# needed for facing, and a nonzero value reads as a sideways head-turn, not
# attention (see history above).
def _look_targets(pan: float, tilt: float) -> dict[str, float]:
    return {
        "base_yaw_joint": _clip(pan, -2.45, 2.45),
        "shoulder_pitch_joint": ATTENTIVE_SHOULDER_PITCH,
        "elbow_pitch_joint": ATTENTIVE_ELBOW_PITCH,
        "neck_yaw_joint": 0.0,
        "head_pitch_joint": _clip(tilt, -0.8, 0.6),
    }


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


HOME_POSE = {
    "base_yaw_joint": 0.0,
    "shoulder_pitch_joint": 0.0,
    "elbow_pitch_joint": 0.0,
    "neck_yaw_joint": 0.0,
    "head_pitch_joint": 0.0,
}

NOD_POSE_DELTA = {"head_pitch_joint": 0.35}
SHAKE_POSE_DELTA = {"neck_yaw_joint": 0.4}

# Idle wandering (nobody engaged): gentle, curious-looking drift, not the
# alert "attentive" lean used for look_at -- the two poses should read as
# different moods. Ranges stay comfortably inside each joint's soft limits.
IDLE_BASE_YAW_RANGE = (-1.2, 1.2)
IDLE_SHOULDER_RANGE = (-0.3, 0.4)
IDLE_ELBOW_RANGE = (-1.2, -0.3)
IDLE_NECK_YAW_RANGE = (-0.6, 0.6)
IDLE_HEAD_PITCH_RANGE = (-0.3, 0.2)
IDLE_SPEED_SCALE = 0.35  # slower than a reflex -- idle drift, not a reaction


@dataclass
class ExecutorHooks:
    """Side effects the executor triggers that aren't simulator state --
    wired up to real implementations later (speech/sfx modules); default to
    no-ops so the body layer is independently testable."""

    on_speak: callable = field(default=lambda text: None)
    on_play_sound: callable = field(default=lambda name: None)
    on_observe: callable = field(default=lambda: None)


class ActionExecutor:
    def __init__(self, sim: LampSimulator, hooks: ExecutorHooks | None = None):
        self.sim = sim
        self.player = TrajectoryPlayer(sim)
        self.hooks = hooks or ExecutorHooks()
        self._telemetry: list[Telemetry] = []

    def run(self, action: Action, speed_scale: float = 1.0) -> None:
        started = time.time()
        self._telemetry.append(Telemetry(kind="action_started", payload={"kind": action.kind}))
        try:
            self._dispatch(action, speed_scale)
        except Exception as exc:  # noqa: BLE001 -- log and keep the demo alive
            self._telemetry.append(
                Telemetry(kind="action_failed", payload={"kind": action.kind, "error": str(exc)})
            )
            return
        self._telemetry.append(
            Telemetry(
                kind="action_done",
                payload={"kind": action.kind, "elapsed_s": time.time() - started},
            )
        )

    def _dispatch(self, action: Action, speed_scale: float) -> None:
        p = action.params
        if action.kind == "look_at":
            self._move_and_settle(_look_targets(p.get("pan", 0.0), p.get("tilt", 0.0)), speed_scale)
        elif action.kind == "point_at":
            self._move_and_settle(_look_targets(p.get("pan", 0.0), p.get("tilt", 0.0)), speed_scale)
        elif action.kind == "nod":
            current = self.sim.get_all_joint_angles()
            up = {"head_pitch_joint": current["head_pitch_joint"] + NOD_POSE_DELTA["head_pitch_joint"]}
            self._move_and_settle(up, speed_scale=1.0)
            self._move_and_settle({"head_pitch_joint": current["head_pitch_joint"]}, speed_scale=1.0)
        elif action.kind == "shake_head":
            current = self.sim.get_all_joint_angles()
            base = current["neck_yaw_joint"]
            for delta in (SHAKE_POSE_DELTA["neck_yaw_joint"], -SHAKE_POSE_DELTA["neck_yaw_joint"], 0.0):
                self._move_and_settle({"neck_yaw_joint": base + delta}, speed_scale=1.0)
        elif action.kind == "home":
            self._move_and_settle(HOME_POSE, speed_scale)
        elif action.kind == "idle_sway":
            targets = {
                "base_yaw_joint": random.uniform(*IDLE_BASE_YAW_RANGE),
                "shoulder_pitch_joint": random.uniform(*IDLE_SHOULDER_RANGE),
                "elbow_pitch_joint": random.uniform(*IDLE_ELBOW_RANGE),
                "neck_yaw_joint": random.uniform(*IDLE_NECK_YAW_RANGE),
                "head_pitch_joint": random.uniform(*IDLE_HEAD_PITCH_RANGE),
            }
            self._move_and_settle(targets, speed_scale=IDLE_SPEED_SCALE)
        elif action.kind == "set_light":
            self.sim.set_light(
                on=p.get("on", True),
                color=tuple(p.get("color", (1.0, 0.95, 0.76))),
                brightness=p.get("brightness", 1.0),
            )
        elif action.kind == "play_sound":
            self.hooks.on_play_sound(p["name"])
        elif action.kind == "speak":
            self.hooks.on_speak(p["text"])
        elif action.kind == "observe":
            self.hooks.on_observe()
        else:
            raise ValueError(f"Unhandled action kind: {action.kind}")

    def _move_and_settle(self, targets: dict[str, float], speed_scale: float) -> None:
        self.player.move_to(targets, speed_scale=speed_scale)
        while self.player.is_moving():
            self.player.step(DT)
            self.sim.forward()
            time.sleep(DT)

    def drain_telemetry(self) -> list[Telemetry]:
        out, self._telemetry = self._telemetry, []
        return out
