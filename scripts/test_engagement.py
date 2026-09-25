"""Live test: opens the lamp in a GUI window and your webcam, and reacts when
you look at the camera. Sit in frame -> lamp turns toward you, nods, and its
light brightens. Look away for ~2 seconds -> it dims and returns home.

Usage: .venv/bin/python scripts/test_engagement.py
Ctrl+C to quit. macOS will prompt for camera permission the first time.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.body.executor import ActionExecutor
from src.body.sim import LampSimulator
from src.character.fsm import CharacterOrchestrator
from src.perception.engagement import EngagementWatcher


def main() -> None:
    sim = LampSimulator(gui=True)
    executor = ActionExecutor(sim)
    watcher = EngagementWatcher()

    print("Starting camera...")
    watcher.start()
    print("Watching. Look at your webcam to engage the lamp; look away to disengage.")

    orchestrator = CharacterOrchestrator(executor, watcher, on_debug=print)
    try:
        orchestrator.run_forever(poll_hz=10.0)
    finally:
        watcher.stop()
        sim.close()


if __name__ == "__main__":
    main()
