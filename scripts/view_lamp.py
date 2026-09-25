"""Interactive PyBullet viewer: opens a real window on your screen so you can
orbit/zoom with the mouse (left-drag orbit, scroll zoom, right/middle-drag
pan) and manually move every joint + the light with on-screen sliders.

Usage: .venv/bin/python scripts/view_lamp.py
Close the window (or Ctrl+C in the terminal) to quit.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pybullet as p

from src.body.sim import CONTROLLED_JOINTS, LampSimulator


def main() -> None:
    sim = LampSimulator(gui=True)

    joint_sliders = {}
    for name in CONTROLLED_JOINTS:
        limits = sim.limits_for(name)
        joint_sliders[name] = p.addUserDebugParameter(name, limits.lower, limits.upper, 0.0)

    light_on_slider = p.addUserDebugParameter("light_on (>0.5)", 0, 1, 1)
    brightness_slider = p.addUserDebugParameter("light_brightness", 0, 1, 0.9)
    hue_slider = p.addUserDebugParameter("light_hue (0=warm .. 1=cool)", 0, 1, 0.0)

    print("Viewer running. Drag the sliders in the window to move the lamp.")
    print("Mouse: left-drag orbit, scroll zoom, right/middle-drag pan.")

    try:
        while True:
            for name, slider_id in joint_sliders.items():
                sim.set_joint_angle(name, p.readUserDebugParameter(slider_id))

            on = p.readUserDebugParameter(light_on_slider) > 0.5
            brightness = p.readUserDebugParameter(brightness_slider)
            hue = p.readUserDebugParameter(hue_slider)
            warm = (1.0, 0.6, 0.2)
            cool = (0.6, 0.75, 1.0)
            color = tuple(warm[i] + (cool[i] - warm[i]) * hue for i in range(3))
            sim.set_light(on=on, color=color, brightness=brightness)

            time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()


if __name__ == "__main__":
    main()
