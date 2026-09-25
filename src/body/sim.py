"""MuJoCo adapter for the dummy 5-DOF lamp.

This module is the *only* place in the codebase that talks to MuJoCo or
knows joint/body names. Everything above it (the executor, the character
brain) only ever deals in the Action vocabulary from `src.protocol.models`.

Joints are driven *kinematically*: we write directly into `data.qpos` each
tick (after velocity-limited interpolation upstream in trajectory.py) and
call `mj_forward` to recompute derived poses, rather than using MuJoCo
actuators + `mj_step` dynamics. The URDF defines no <transmission>/actuators
(by design -- see its header comment), so there is nothing for a PD
controller to drive anyway. Kinematic driving is also simpler to reason
about for a gesture-demo lamp than tuning real dynamics, and we still
enforce the URDF's own velocity/effort limits ourselves (see JointLimits).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco

URDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "robot", "dummy_lamp_5dof.urdf"
)
# Relative to the URDF's own directory (matches the URDF's own <mesh> reference).
SHADE_MESH_FILE = "assets/lamp_shade.stl"

CONTROLLED_JOINTS = [
    "base_yaw_joint",
    "shoulder_pitch_joint",
    "elbow_pitch_joint",
    "neck_yaw_joint",
    "head_pitch_joint",
]

HEAD_BODY_NAME = "head_gimbal_link"

# MuJoCo's URDF importer keeps <limit lower="..." upper="..."> but has no
# concept of joint velocity/effort (those come from URDF <transmission>,
# which this fixture deliberately omits). We keep the URDF's own numbers
# here, read once by hand from dummy_lamp_5dof.urdf, and enforce them in
# trajectory.py's interpolator instead of via a PD controller.
VELOCITY_LIMITS_RAD_S = {
    "base_yaw_joint": 1.50,
    "shoulder_pitch_joint": 0.95,
    "elbow_pitch_joint": 1.15,
    "neck_yaw_joint": 1.60,
    "head_pitch_joint": 1.45,
}
EFFORT_LIMITS_NM = {
    "base_yaw_joint": 6.0,
    "shoulder_pitch_joint": 12.0,
    "elbow_pitch_joint": 9.0,
    "neck_yaw_joint": 3.5,
    "head_pitch_joint": 3.2,
}


@dataclass
class JointLimits:
    lower: float
    upper: float
    velocity: float
    effort: float


class LampSimulator:
    """Thin wrapper around a MuJoCo model/data pair holding the lamp URDF."""

    def __init__(self, urdf_path: str = URDF_PATH):
        spec = mujoco.MjSpec.from_file(urdf_path)
        self._augment_visuals(spec)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self._qpos_adr: dict[str, int] = {}
        self._limits: dict[str, JointLimits] = {}
        for name in CONTROLLED_JOINTS:
            joint = self.model.joint(name)
            self._qpos_adr[name] = int(joint.qposadr[0])
            lower, upper = joint.range
            self._limits[name] = JointLimits(
                lower=float(lower),
                upper=float(upper),
                velocity=VELOCITY_LIMITS_RAD_S[name],
                effort=EFFORT_LIMITS_NM[name],
            )

        self._light_geom_id = self._find_light_geom_id()
        self._resize_light_geom()

        for name in CONTROLLED_JOINTS:
            self.data.qpos[self._qpos_adr[name]] = 0.0
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _augment_visuals(spec: "mujoco.MjSpec") -> None:
        """Add back the lamp-shade mesh, which MuJoCo's default URDF import
        drops (it keeps only collision primitives -- see NOTES.md), so the
        demo recording reads as a lamp rather than a bare mechanical arm."""
        spec.add_mesh(name="lamp_shade", file=SHADE_MESH_FILE)
        head = spec.body(HEAD_BODY_NAME)
        shade = head.add_geom(
            name="lamp_shade_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="lamp_shade",
            pos=[0.070, 0, 0],
        )
        shade.rgba = [0.94, 0.94, 0.95, 1.0]
        # NOTE: setting shade.contype/conaffinity here -- even just to mark
        # this visual-only mesh as non-colliding -- makes MjSpec.compile()
        # silently discard *all* spec edits on mujoco 3.14, with no
        # exception raised (reproduced independently of geom type). Left at
        # MuJoCo's collision defaults; harmless for this fixed-base,
        # slow-motion demo with nothing else nearby to collide with.

    def _find_light_geom_id(self) -> int:
        """The 'light' is faked by recoloring the geom MuJoCo's URDF importer
        colored with the "fixture_light" material (rgba alpha 0.95 -- the
        only material in the URDF using a non-1.0 alpha, so it's a reliable,
        semantic way to find it rather than guessing by geom order)."""
        head_id = self.model.body(HEAD_BODY_NAME).id
        candidates = [
            i
            for i in range(self.model.ngeom)
            if self.model.geom_bodyid[i] == head_id and self.model.geom_rgba[i][3] < 1.0
        ]
        if not candidates:
            raise RuntimeError("Could not find the head geom to use as the light indicator")
        return candidates[0]

    def _resize_light_geom(self) -> None:
        """MuJoCo's URDF import gave the light-material geom the size of the
        *whole head collision cylinder*, which fully occludes the shade mesh
        sitting at the same spot. Shrink and reposition it, post-compile, to
        sit as a rounded bulb just inside the shade's mouth (the mesh's own
        outer bound is ~0.137 along local x from head_gimbal_link).

        A first attempt used the URDF's literal light-disc dimensions
        (r=0.090, half-length=0.003 -- a paper-thin disc). That geom *was*
        rendering correctly (verified by temporarily blowing it up to a giant
        red cylinder), it was just nearly invisible edge-on: a sub-millimeter
        cylinder reduces to a thin line from most camera angles. A rounder
        bulb shape (comparable radius and half-length) reads as "a glowing
        bulb" from far more viewing angles, closer to the reference render.

        This is a plain model-array mutation, not a spec edit, so it doesn't
        touch the add_geom/compile bug noted above."""
        self.model.geom_size[self._light_geom_id] = [0.055, 0.045, 0.0]
        self.model.geom_pos[self._light_geom_id] = [0.20, 0.0, 0.0]

    # -- joint control -----------------------------------------------------

    def limits_for(self, joint_name: str) -> JointLimits:
        return self._limits[joint_name]

    def set_joint_angle(self, joint_name: str, angle: float) -> None:
        """Set a joint's position for this instant, clamped to the URDF's own
        limits. Velocity limiting happens one level up in trajectory.py's
        interpolator -- this call always takes the angle it's given as truth."""
        limits = self._limits[joint_name]
        clamped = max(limits.lower, min(limits.upper, angle))
        self.data.qpos[self._qpos_adr[joint_name]] = clamped

    def get_joint_angle(self, joint_name: str) -> float:
        return float(self.data.qpos[self._qpos_adr[joint_name]])

    def get_all_joint_angles(self) -> dict[str, float]:
        return {name: self.get_joint_angle(name) for name in CONTROLLED_JOINTS}

    # -- light (faked -- URDF defines no real light behavior) ---------------

    def set_light(self, on: bool, color: tuple[float, float, float] = (1.0, 0.95, 0.76), brightness: float = 1.0) -> None:
        if on:
            r, g, b = color
            rgba = [r * brightness, g * brightness, b * brightness, 1.0]
        else:
            rgba = [0.15, 0.15, 0.15, 1.0]
        self.model.geom_rgba[self._light_geom_id] = rgba

    # -- lifecycle -----------------------------------------------------------

    def forward(self) -> None:
        """Recompute derived kinematics (body/geom world poses) after directly
        setting qpos. No mj_step / dynamics integration -- joints are driven
        kinematically, so there's nothing to integrate."""
        mujoco.mj_forward(self.model, self.data)
