"""Manual smoke test for the body layer: sweep the lamp through a few poses,
toggle the light, and save frames so we can actually see it move.

Usage: .venv/bin/python scripts/test_sim.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mujoco
import numpy as np
from PIL import Image

from src.body.sim import LampSimulator
from src.body.trajectory import DT, TrajectoryPlayer

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "scratch_renders")
os.makedirs(OUT_DIR, exist_ok=True)

POSES = [
    ("home", {}),
    ("lean_and_look_down", {"shoulder_pitch_joint": 0.7, "elbow_pitch_joint": -1.2, "head_pitch_joint": 0.3}),
    ("look_left_up", {"base_yaw_joint": 0.9, "head_pitch_joint": -0.5, "neck_yaw_joint": 0.3}),
    ("look_right_down", {"base_yaw_joint": -0.9, "head_pitch_joint": 0.4, "neck_yaw_joint": -0.3}),
    ("nod_down", {"head_pitch_joint": 0.6}),
    ("nod_up", {"head_pitch_joint": -0.3}),
    ("home2", {"base_yaw_joint": 0.0, "shoulder_pitch_joint": 0.0, "elbow_pitch_joint": 0.0, "neck_yaw_joint": 0.0, "head_pitch_joint": 0.0}),
]


def render(sim: LampSimulator, renderer: "mujoco.Renderer", tag: str) -> None:
    renderer.update_scene(sim.data, camera=-1)
    img = renderer.render()
    Image.fromarray(img).save(os.path.join(OUT_DIR, f"{tag}.png"))


def main() -> None:
    sim = LampSimulator()
    player = TrajectoryPlayer(sim)
    renderer = mujoco.Renderer(sim.model, height=480, width=640)

    render(sim, renderer, "00_initial")

    for i, (name, targets) in enumerate(POSES, start=1):
        sim.set_light(on=(i % 2 == 0), color=(1.0, 0.6, 0.2), brightness=0.9)
        player.move_to(targets)
        steps = 0
        while player.is_moving() and steps < 2000:
            player.step(DT)
            sim.forward()
            steps += 1
        print(f"{i:02d} {name}: reached {sim.get_all_joint_angles()} in {steps} steps")
        render(sim, renderer, f"{i:02d}_{name}")

    print(f"Saved frames to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
