"""PyBullet adapter for the dummy 5-DOF lamp.

This module is the *only* place in the codebase that talks to PyBullet or
knows joint/link names. Everything above it (the executor, the character
brain) only ever deals in the Action vocabulary from `src.protocol.models`.

Unlike MuJoCo's URDF importer (which keeps only one collision-derived geom
per link and drops mesh visuals -- see git history for the workarounds that
required), PyBullet's URDF loader renders every <visual> tag faithfully,
including the lamp-shade mesh, with no hand-authored color or geometry
patches. It only needed the shade STL converted from ASCII to binary (see
scripts/stl_ascii_to_binary.py) since neither engine's mesh loader parses
ASCII STL.

Joints are driven *kinematically*: we teleport joint angles directly via
resetJointState each tick (after velocity-limited interpolation upstream in
trajectory.py), rather than using PyBullet's position-control motors +
stepSimulation dynamics. The URDF defines no <transmission>/actuators (by
design -- see its header comment), so there's nothing for a PD controller to
drive anyway, and kinematic driving is simpler to reason about for a
gesture-demo lamp than tuning real dynamics. We still enforce the URDF's own
velocity/effort limits ourselves (see JointLimits).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pybullet as p

URDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "robot", "dummy_lamp_5dof.urdf"
)

CONTROLLED_JOINTS = [
    "base_yaw_joint",
    "shoulder_pitch_joint",
    "elbow_pitch_joint",
    "neck_yaw_joint",
    "head_pitch_joint",
]

LIGHT_LINK_NAME = "light_emitter_link"


@dataclass
class JointLimits:
    lower: float
    upper: float
    velocity: float
    effort: float


class LampSimulator:
    """Thin wrapper around a PyBullet instance holding the lamp URDF."""

    def __init__(self, gui: bool = False, urdf_path: str = URDF_PATH):
        self._client = p.connect(p.GUI if gui else p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)

        # useFixedBase=True: the lamp is bolted to a desk, not free-floating.
        self.body_id = p.loadURDF(
            urdf_path, basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=self._client
        )

        if gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.8,
                cameraYaw=35,
                cameraPitch=-15,
                cameraTargetPosition=[0, 0, 0.3],
                physicsClientId=self._client,
            )

        self._joint_index: dict[str, int] = {}
        self._link_index: dict[str, int] = {}
        self._limits: dict[str, JointLimits] = {}
        self._discover_joints_and_links()
        self._hide_semantic_markers()

        for name in CONTROLLED_JOINTS:
            self.set_joint_angle(name, 0.0)

    def _discover_joints_and_links(self) -> None:
        n = p.getNumJoints(self.body_id, physicsClientId=self._client)
        for i in range(n):
            info = p.getJointInfo(self.body_id, i, physicsClientId=self._client)
            joint_name = info[1].decode("utf-8")
            child_link_name = info[12].decode("utf-8")
            self._link_index[child_link_name] = i  # PyBullet: link i is fixed to joint i

            if joint_name in CONTROLLED_JOINTS:
                self._joint_index[joint_name] = i
                lower, upper = info[8], info[9]
                self._limits[joint_name] = JointLimits(
                    lower=lower, upper=upper, velocity=info[11], effort=info[10]
                )

        missing = set(CONTROLLED_JOINTS) - set(self._joint_index)
        if missing:
            raise RuntimeError(f"URDF is missing expected joints: {missing}")
        if LIGHT_LINK_NAME not in self._link_index:
            raise RuntimeError(f"URDF is missing expected link: {LIGHT_LINK_NAME}")

    def _hide_semantic_markers(self) -> None:
        """camera_link and speaker_link are explicitly "semantic frames" per
        the URDF's own comment -- markers for where a real camera/speaker
        would sit, not something with real rendered behavior. We use the
        laptop's actual camera/mic for perception (per the challenge), so
        these markers have no function here; left visible, camera_link in
        particular renders as a distracting black dot right next to the
        shade opening. Fade them into the body's own coloring instead of
        deleting them, so the link/joint structure stays intact."""
        for link_name in ("camera_link", "speaker_link"):
            if link_name in self._link_index:
                p.changeVisualShape(
                    self.body_id,
                    self._link_index[link_name],
                    rgbaColor=[0.66, 0.68, 0.71, 1.0],
                    physicsClientId=self._client,
                )

    # -- joint control -----------------------------------------------------

    def limits_for(self, joint_name: str) -> JointLimits:
        return self._limits[joint_name]

    def set_joint_angle(self, joint_name: str, angle: float) -> None:
        """Set a joint's position for this instant, clamped to the URDF's own
        limits. Velocity limiting happens one level up in trajectory.py's
        interpolator -- this call always takes the angle it's given as truth."""
        limits = self._limits[joint_name]
        clamped = max(limits.lower, min(limits.upper, angle))
        p.resetJointState(
            self.body_id, self._joint_index[joint_name], clamped, physicsClientId=self._client
        )

    def get_joint_angle(self, joint_name: str) -> float:
        state = p.getJointState(self.body_id, self._joint_index[joint_name], physicsClientId=self._client)
        return state[0]

    def get_all_joint_angles(self) -> dict[str, float]:
        return {name: self.get_joint_angle(name) for name in CONTROLLED_JOINTS}

    # -- light (faked -- URDF defines no real light behavior) ---------------

    def set_light(self, on: bool, color: tuple[float, float, float] = (1.0, 0.95, 0.76), brightness: float = 1.0) -> None:
        link_idx = self._link_index[LIGHT_LINK_NAME]
        if on:
            r, g, b = color
            rgba = [r * brightness, g * brightness, b * brightness, 1.0]
        else:
            rgba = [0.15, 0.15, 0.15, 1.0]
        p.changeVisualShape(self.body_id, link_idx, rgbaColor=rgba, physicsClientId=self._client)

    # -- rendering -----------------------------------------------------------

    def render(
        self,
        width: int = 640,
        height: int = 480,
        distance: float = 1.8,
        yaw: float = 35,
        pitch: float = -15,
        target: tuple[float, float, float] = (0.0, 0.0, 0.3),
    ):
        """Headless RGB render (works identically with or without a GUI
        window -- used both for demo screenshots and for the eventual
        Ubuntu deployment, which has no guaranteed display). Defaults match
        the GUI debug camera set in __init__."""
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=list(target),
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self._client,
        )
        proj = p.computeProjectionMatrixFOV(fov=45, aspect=width / height, nearVal=0.05, farVal=3.0)
        _, _, rgba, _, _ = p.getCameraImage(
            width, height, viewMatrix=view, projectionMatrix=proj, physicsClientId=self._client
        )
        import numpy as np

        return np.reshape(rgba, (height, width, 4))[:, :, :3]

    # -- lifecycle -----------------------------------------------------------

    def forward(self) -> None:
        """No-op: joints are driven kinematically via resetJointState, which
        takes effect immediately -- there's no separate dynamics step to
        advance. Kept so trajectory.py/executor.py don't need to know which
        physics engine is underneath."""

    def close(self) -> None:
        p.disconnect(physicsClientId=self._client)
