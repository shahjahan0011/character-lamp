"""Live test: opens the lamp in a GUI window and your webcam, and reacts when
you look at the camera. Sit in frame -> music stops, a chime plays, it turns
toward you, nods, and flashes then brightens. Look away for ~2 seconds ->
it dims, returns home, and the idle music resumes.

Usage: .venv/bin/python scripts/test_engagement.py
Ctrl+C to quit. macOS will prompt for camera and microphone permission the
first time (microphone isn't used yet, but sounddevice may still ask).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.body.executor import ActionExecutor, ExecutorHooks
from src.body.sim import LampSimulator
from src.character.fsm import CharacterOrchestrator
from src.perception.engagement import EngagementWatcher
from src.speech.playback import make_audio_hooks


def main() -> None:
    sim = LampSimulator(gui=True, clean_gui=True)
    executor = ActionExecutor(sim, hooks=ExecutorHooks(**make_audio_hooks()))
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
