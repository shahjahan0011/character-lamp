"""Primary launcher for the character-lamp demo.

Opens the lamp in a GUI window (or headless), your webcam, and your
microphone. Sit in frame -> music stops, a chime plays, it turns toward
you, nods, flashes then brightens, and starts listening. Say something ->
local silence detection commits your utterance to a persistent Gemini
Live session; a short "thinking" cue plays until the reply's audio starts
arriving, then it speaks while reacting with whatever gesture it calls.
Ask it to remember an object, recall what it's seen, or find something and
point at it -- see docs/DEMO_SCRIPT.md for a scripted walkthrough of all
five challenge moments in one continuous interaction.

Usage:
  python scripts/run_character.py                  # normal run
  python scripts/run_character.py --headless        # no GUI window
  python scripts/run_character.py --offline          # deterministic mock demo, no hardware/network
  python scripts/run_character.py --list-audio-devices
  python scripts/run_character.py --force-engaged    # real hardware, but skip face-detection gating

Requires GEMINI_API_KEY in a local .env (copy .env.example, add your key
from https://aistudio.google.com/apikey -- no credit card needed) for
conversation, scene memory, and goal-directed action. Engagement, motion,
light, and music all work without a key.

Ctrl+C to quit. macOS/Ubuntu will prompt for camera and microphone
permission the first time.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

from src.body.executor import ActionExecutor, ExecutorHooks
from src.body.sim import LampSimulator
from src.character.fsm import CharacterOrchestrator
from src.character.memory import SceneMemory
from src.perception.engagement import EngagementState, EngagementWatcher
from src.protocol.observation import DetectedObject, Observation
from src.speech.audio_mixer import AudioMixer
from src.speech.error_speech import ErrorSpeech
from src.speech.live_capture import LiveMicStreamer
from src.speech.live_client import LIVE_MODEL, OUTPUT_SAMPLE_RATE, GeminiLiveClient
from src.speech.playback import make_audio_hooks

log = logging.getLogger("character_lamp")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument(
        "--list-audio-devices", action="store_true", help="List available audio input/output devices and exit"
    )
    parser.add_argument("--headless", action="store_true", help="Run PyBullet without a GUI window")
    parser.add_argument(
        "--offline", "--mock-model", dest="offline", action="store_true",
        help="Deterministic mock demo: no camera/mic/network, exercises the real goal/memory/tool workflow with scripted vision results",
    )
    parser.add_argument(
        "--force-engaged", action="store_true",
        help="Diagnostic: use the real camera but always report engaged=True, skipping face-detection gating",
    )
    parser.add_argument("--data-dir", type=str, default="var", help="Directory for persisted scene memory (default: var)")
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level for this launcher's own status messages (default: INFO)",
    )
    return parser.parse_args(argv)


def list_audio_devices() -> None:
    import sounddevice as sd

    print(sd.query_devices())


class _ForceEngagedWatcher:
    """Wraps a real EngagementWatcher (real camera, real frames) but always
    reports engaged=True -- a diagnostic bypass for the face-detection
    gate specifically, not a full mock (the camera and frames are real)."""

    def __init__(self, watcher: EngagementWatcher):
        self._watcher = watcher

    def start(self) -> None:
        self._watcher.start()

    def stop(self) -> None:
        self._watcher.stop()

    def get_state(self) -> EngagementState:
        return EngagementState(engaged=True, face_x_frac=0.0, changed_at=time.time())

    def get_latest_frame(self):
        return self._watcher.get_latest_frame()


def _wait_for_observation(orchestrator: CharacterOrchestrator, timeout_s: float = 5.0):
    """Waits for SceneObserver to finish its current job and returns the
    result. A short initial delay first: is_busy() can read False before
    the background worker thread has even picked up the just-submitted
    job, which would make a naive wait-loop return before real processing
    happens."""
    time.sleep(0.05)
    deadline = time.time() + timeout_s
    while orchestrator.observer.is_busy() and time.time() < deadline:
        time.sleep(0.02)
    return orchestrator.observer.pop_new_observation()


def run_offline_demo(headless: bool, data_dir: Path) -> None:
    """Deterministic, hardware/network-free demonstration -- exercises the
    exact same CharacterOrchestrator/ToolGateway/GoalCoordinator/
    SceneMemory/ActionExecutor classes as the live path (not a separate
    fake-success path); only the outbound vision call is replaced with a
    scripted, deterministic Observation so this can run with no camera,
    microphone, speaker, display (if --headless), or API key."""
    log.info("Running offline/mock demo (no camera, mic, network, or API key needed)")

    scripted_observations = [
        Observation(
            purpose="object_memory", width=640, height=480,
            objects=[DetectedObject(label="Red mug", color="red", attributes="ceramic, on the desk", image_x=0.62, image_y=0.55, confidence=0.95)],
        ),
        Observation(
            purpose="goal_planning", width=640, height=480,
            objects=[DetectedObject(label="Red mug", color="red", attributes="ceramic, on the desk", image_x=0.62, image_y=0.55, confidence=0.93)],
        ),
        Observation(
            purpose="goal_verification", width=640, height=480,
            objects=[DetectedObject(label="Red mug", color="red", attributes="ceramic, on the desk", image_x=0.62, image_y=0.55, confidence=0.93)],
        ),
    ]
    call_count = {"n": 0}

    def fake_describe_scene(frame, purpose="scene", timeout_s=30.0):
        idx = min(call_count["n"], len(scripted_observations) - 1)
        call_count["n"] += 1
        obs = scripted_observations[idx]
        # Fresh id AND a fresh capture timestamp -- model_copy() alone
        # would keep the template's original captured_at_monotonic (set
        # once, when scripted_observations was built), which is exactly
        # what GoalCoordinator's freshness/ordering checks correctly
        # reject a real stale observation for.
        return obs.model_copy(update={"purpose": purpose, "observation_id": uuid.uuid4().hex, "captured_at_monotonic": time.monotonic()})

    sim = LampSimulator(gui=not headless, clean_gui=True)
    mixer = AudioMixer(sample_rate=OUTPUT_SAMPLE_RATE)
    if not headless:
        mixer.start()
    executor = ActionExecutor(sim, hooks=ExecutorHooks(**mixer.make_hooks()))
    fake_frame = __import__("numpy").zeros((480, 640, 3), dtype="uint8")

    class _FakeWatcher:
        def get_state(self):
            return EngagementState(engaged=True, face_x_frac=0.0, changed_at=time.time())

        def get_latest_frame(self):
            return fake_frame

    memory = SceneMemory(persist_path=data_dir / "scene_memory.json", on_debug=log.debug)

    with patch("src.speech.vision.describe_scene", side_effect=fake_describe_scene):
        orchestrator = CharacterOrchestrator(
            executor, _FakeWatcher(), audio_mixer=mixer, memory=memory, on_debug=log.info
        )
        orchestrator.tick()  # ENGAGE -- also fires an ambient "scene" observation
        log.info("[1/8] engagement transition -> ENGAGE")
        _wait_for_observation(orchestrator)
        orchestrator.observer.pop_new_observation()  # drain the ambient one; not part of the scripted steps

        log.info("[2/8] scene observation (object_memory)")
        result = orchestrator.gateway.execute("request_observation", {"purpose": "object_memory"}, call_id="mock-1")
        assert result is None, "request_observation should queue asynchronously"
        obs_result = _wait_for_observation(orchestrator)
        assert obs_result is not None, "offline demo: scripted observation did not complete"
        observation_id = obs_result.observation.observation_id

        log.info("[3/8] memory write (remember_object)")
        remember_result = orchestrator.gateway.execute(
            "remember_object",
            {
                "observation_id": observation_id, "label": "Red mug", "color": "red",
                "attributes": "ceramic, on the desk", "image_x": 0.62, "image_y": 0.55,
                "confidence": 0.95, "notes": None,
            },
            call_id="mock-2",
        )
        assert remember_result["ok"] is True
        log.info(f"    memory now has {len(memory)} record(s)")

        log.info("[4/8] memory recall (recall_memory)")
        recall_result = orchestrator.gateway.execute("recall_memory", {"query": "mug", "limit": 5}, call_id="mock-3")
        assert recall_result["ok"] is True and len(recall_result["records"]) == 1
        log.info(f"    recalled: {recall_result['records'][0]['label']}")

        log.info("[5/8] goal planning (request_observation -> goal_planning)")
        orchestrator.gateway.execute("request_observation", {"purpose": "goal_planning"}, call_id="mock-4")
        obs_result = _wait_for_observation(orchestrator)
        assert obs_result is not None, "offline demo: goal_planning observation did not complete"
        orchestrator.gateway.complete_observation_for_goal(obs_result.observation.observation_id, "goal_planning")
        goal_id = orchestrator.gateway._current_goal_id
        planning_obs_id = obs_result.observation.observation_id
        assert orchestrator.goals.get(goal_id).stage == "planning"

        log.info("[6/8] bounded pointing action (look_at_image_point)")
        point_result = orchestrator.gateway.execute(
            "look_at_image_point", {"observation_id": planning_obs_id, "x": 0.62, "y": 0.55}, call_id="mock-5"
        )
        assert point_result["ok"] is True
        assert orchestrator.goals.get(goal_id).stage == "acting"
        log.info(f"    pan={point_result['pan']:.2f} tilt={point_result['tilt']:.2f}")

        log.info("[7/8] re-observation (request_observation -> goal_verification)")
        orchestrator.gateway.execute("request_observation", {"purpose": "goal_verification"}, call_id="mock-6")
        obs_result = _wait_for_observation(orchestrator)
        assert obs_result is not None, "offline demo: goal_verification observation did not complete"
        orchestrator.gateway.complete_observation_for_goal(obs_result.observation.observation_id, "goal_verification")
        verify_obs_id = obs_result.observation.observation_id
        assert orchestrator.goals.get(goal_id).stage == "reobserving"

        log.info("[8/8] goal completion (finish_goal) -- success and failure paths")
        success_result = orchestrator.gateway.execute(
            "finish_goal",
            {"goal_id": goal_id, "post_action_observation_id": verify_obs_id, "success": True,
             "evidence": "mug visible at the pointed location", "confidence": 0.9},
            call_id="mock-7",
        )
        assert success_result["ok"] is True and success_result["stage"] == "verified"
        log.info(f"    SUCCESS case: goal stage = {success_result['stage']}")

        # Also demonstrate the failure/invalid-ordering path -- a bare
        # finish_goal with no workflow behind it is rejected with a
        # structured error, not silently accepted.
        rejected = orchestrator.gateway.execute(
            "finish_goal",
            {"goal_id": "never-started", "post_action_observation_id": "x", "success": True,
             "evidence": "e", "confidence": 0.9},
            call_id="mock-8",
        )
        assert rejected["ok"] is False
        log.info(f"    FAILURE case (invalid ordering): rejected as expected -- {rejected['error']}")

    orchestrator.observer.stop()
    mixer.close()
    sim.close()
    log.info("Offline demo complete -- all 8 steps passed using the real goal/memory/tool workflow.")


def run_live(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log.warning(
            "No GEMINI_API_KEY found -- conversation, scene memory, and goal-directed "
            "action will be disabled (add .env to enable). Engagement, motion, light, "
            "and music all work without a key."
        )

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    sim = LampSimulator(gui=not args.headless, clean_gui=True)
    mixer = AudioMixer(sample_rate=OUTPUT_SAMPLE_RATE)
    mixer.start()
    hooks_dict = make_audio_hooks(speak=bool(api_key))
    hooks_dict.update(mixer.make_hooks())
    executor = ActionExecutor(sim, hooks=ExecutorHooks(**hooks_dict))

    real_watcher = EngagementWatcher(camera_index=args.camera_index)
    watcher = _ForceEngagedWatcher(real_watcher) if args.force_engaged else real_watcher

    live_client = None
    mic_streamer = None
    error_speech = None
    if api_key:
        live_client = GeminiLiveClient(LIVE_MODEL, api_key, on_audio_chunk=mixer.enqueue_speech_pcm16, on_debug=log.debug)
        mic_streamer = LiveMicStreamer(live_client, on_debug=log.debug)
        error_speech = ErrorSpeech(mixer, on_debug=log.debug)
        error_speech.preload()

    log.info("Starting camera and microphone...")
    real_watcher.start()
    if mic_streamer is not None:
        mic_streamer.start()
        log.info("Connecting to Gemini Live...")
        if live_client.wait_until_connected(timeout_s=15.0):
            log.info("Connected.")
        else:
            log.warning("Still connecting in the background (will keep retrying)...")
    if args.force_engaged:
        log.warning("--force-engaged: face-detection gating is BYPASSED (always engaged)")
    log.info("Watching. Look at your webcam to engage the lamp; look away to disengage.")
    log.info("While engaged, speak -- it only listens while it's paying attention to you.")

    memory = SceneMemory(persist_path=data_dir / "scene_memory.json", on_debug=log.debug)
    orchestrator = CharacterOrchestrator(
        executor, watcher, live_client=live_client, audio_mixer=mixer, memory=memory,
        error_speech=error_speech, on_debug=log.debug,
    )
    try:
        orchestrator.run_forever(poll_hz=10.0)
    finally:
        log.info("Shutting down...")
        real_watcher.stop()
        orchestrator.observer.stop()
        if mic_streamer is not None:
            mic_streamer.stop()
        if live_client is not None:
            live_client.close()
        mixer.close()
        sim.close()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")

    if args.list_audio_devices:
        list_audio_devices()
        return
    if args.offline:
        run_offline_demo(headless=args.headless, data_dir=Path(args.data_dir))
        return
    run_live(args)


if __name__ == "__main__":
    main()
