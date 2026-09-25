"""Live test: opens the lamp in a GUI window, your webcam, and your
microphone. Sit in frame -> music stops, a chime plays, it turns toward
you, nods, flashes then brightens, and starts listening. Say something ->
it transcribes, replies out loud, and speaks through your speaker. Look
away for ~2 seconds -> it stops listening, dims, returns home, and the
idle music resumes.

Requires GEMINI_API_KEY in a local .env (copy .env.example, add your key
from https://aistudio.google.com/apikey -- no credit card needed) for the
transcription/reply/speech steps. Engagement, motion, light, and music
all work without a key.

Usage: .venv/bin/python scripts/test_engagement.py
Ctrl+C to quit. macOS will prompt for camera and microphone permission the
first time.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

from src.body.executor import ActionExecutor, ExecutorHooks
from src.body.sim import LampSimulator
from src.character.fsm import CharacterOrchestrator
from src.perception.engagement import EngagementWatcher
from src.speech.capture import SpeechCapture
from src.speech.playback import make_audio_hooks


def main() -> None:
    load_dotenv()
    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    if not has_key:
        print("No GEMINI_API_KEY found -- speech will listen and log utterances,")
        print("but transcription/reply/speaking will fail (add .env to enable).")

    sim = LampSimulator(gui=True, clean_gui=True)
    executor = ActionExecutor(sim, hooks=ExecutorHooks(**make_audio_hooks(speak=has_key)))
    watcher = EngagementWatcher()
    speech = SpeechCapture()

    print("Starting camera and microphone...")
    watcher.start()
    speech.start()
    print("Watching. Look at your webcam to engage the lamp; look away to disengage.")
    print("While engaged, speak -- it only listens while it's paying attention to you.")

    orchestrator = CharacterOrchestrator(executor, watcher, speech=speech, on_debug=print)
    try:
        orchestrator.run_forever(poll_hz=10.0)
    finally:
        watcher.stop()
        speech.stop()
        sim.close()


if __name__ == "__main__":
    main()
