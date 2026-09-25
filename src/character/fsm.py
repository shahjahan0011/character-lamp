"""Ties perception signals to body actions.

Handles engagement (demo moments 1 and 2: notice someone, acknowledge them,
dim back down when they leave) and spoken interaction (demo moment 3: only
listens while engaged, understands, replies). Memory/goal states get added
here later without changing how this part works -- it just reads whatever
EngagementWatcher/SpeechCapture/DialogueWorker report and reacts.

Deliberately not calling the LLM for the engagement *reaction* (see design
discussion): it needs to feel instantaneous, and a network round-trip would
undercut that, so the acknowledgment is a fixed, hardcoded Action sequence.
The spoken reply, by contrast, has to go through the LLM -- there's no
faking "understood what you said and answered it". That round trip is slow
(measured live: 30-60+ seconds combined), which is exactly why it runs on
DialogueWorker's own background thread rather than inline here -- an
earlier version blocked tick() directly on those calls, which froze
engagement detection (confirmed live: disengage stopped working) for the
entire wait.
"""

from __future__ import annotations

import random
import time
from typing import Callable, Optional

from src.body.executor import ActionExecutor
from src.character.dialogue_worker import DialogueWorker
from src.perception.engagement import EngagementWatcher
from src.protocol.models import Action
from src.speech.capture import SpeechCapture

MAX_PAN_RAD = 0.7  # radians; matches base_yaw_joint's usable range for a look

WARM_WHITE = [1.0, 0.95, 0.76]
NOTICE_FLASH_COUNT = 2
NOTICE_FLASH_INTERVAL_S = 0.12
IDLE_BRIGHTNESS = 0.2
ENGAGED_BRIGHTNESS = 1.0
# Free-tier Gemini audio calls are genuinely slow (measured live: 30-60+
# seconds combined) -- a cool, dim, distinct color, a repeated gentle
# "pondering" gesture, and a soft hum right as it starts, so the wait
# reads as active thinking, not frozen/broken.
THINKING_COLOR = [0.55, 0.7, 1.0]
THINKING_BRIGHTNESS = 0.45
THINK_PULSE_INTERVAL_S = 3.5

# While nobody's engaged, the lamp wanders on its own every so often --
# otherwise it just sits frozen, which reads as "off" rather than "alive but
# not paying attention to anyone right now". The moment someone engages,
# this stops entirely and the lamp focuses on them instead (see tick()).
IDLE_WANDER_MIN_INTERVAL_S = 4.0
IDLE_WANDER_MAX_INTERVAL_S = 9.0


class CharacterOrchestrator:
    def __init__(
        self,
        executor: ActionExecutor,
        watcher: EngagementWatcher,
        speech: Optional[SpeechCapture] = None,
        dialogue_worker: Optional[DialogueWorker] = None,
        on_debug: Optional[Callable[[str], None]] = None,
    ):
        self.executor = executor
        self.watcher = watcher
        self.speech = speech
        # Optional hook for tests/scripts to print what's happening -- the
        # executor swallows action exceptions into telemetry by design (one
        # bad action shouldn't kill the demo), which otherwise means a
        # failure looks identical to "nothing happened". This surfaces it.
        self._on_debug = on_debug or (lambda msg: None)
        self.dialogue_worker = dialogue_worker or (
            DialogueWorker(on_debug=self._on_debug) if speech is not None else None
        )
        self._last_engaged = False
        self._next_idle_wander_at = self._schedule_next_idle_wander()
        self._next_think_pulse_at = 0.0
        # Starts disengaged, so the idle music starts playing immediately.
        self.executor.run(Action(kind="music_on", params={}))

    def _schedule_next_idle_wander(self) -> float:
        return time.time() + random.uniform(IDLE_WANDER_MIN_INTERVAL_S, IDLE_WANDER_MAX_INTERVAL_S)

    def run_forever(self, poll_hz: float = 10.0) -> None:
        period = 1.0 / poll_hz
        try:
            while self.executor.sim.is_connected():
                self.tick()
                time.sleep(period)
            self._on_debug("Body simulator disconnected (window closed?) -- exiting.")
        except KeyboardInterrupt:
            pass

    def tick(self) -> None:
        """Check the current engagement state once and react to a
        just-happened transition. Exposed separately from run_forever so
        tests (and later, a bigger orchestrator loop covering goals) can
        call it directly instead of only via a blocking loop."""
        state = self.watcher.get_state()
        if state.engaged and not self._last_engaged:
            self._on_debug(f"ENGAGE face_x_frac={state.face_x_frac:.2f}")
            self._on_engage(state.face_x_frac)
        elif not state.engaged and self._last_engaged:
            self._on_debug("DISENGAGE")
            self._on_disengage()
        elif not state.engaged and time.time() >= self._next_idle_wander_at:
            self._on_debug("idle wander")
            self.executor.run(Action(kind="idle_sway", params={}))
            self._next_idle_wander_at = self._schedule_next_idle_wander()
        elif state.engaged and self.speech is not None:
            self._check_speech()
        self._last_engaged = state.engaged
        # Checked every tick regardless of the branch above -- a reply can
        # become ready at any moment, independent of whatever else is
        # happening (including a disengage that happened while it was
        # still in flight; see _check_dialogue_reply).
        self._check_dialogue_reply(state.engaged)
        self._check_thinking_pulse()
        self._report_failures()

    def _check_speech(self) -> None:
        if self.dialogue_worker is None:
            return
        utterance = self.speech.get_pending_utterance()
        if utterance is None:
            return
        if self.dialogue_worker.submit(utterance):
            self.executor.run(
                Action(
                    kind="set_light",
                    params={"on": True, "color": THINKING_COLOR, "brightness": THINKING_BRIGHTNESS},
                )
            )
            self.executor.run(Action(kind="play_sound", params={"name": "thinking_hum.wav"}))
            self._next_think_pulse_at = time.time() + THINK_PULSE_INTERVAL_S

    def _check_thinking_pulse(self) -> None:
        """A slow, repeated 'pondering' dip while a reply is in flight --
        otherwise the lamp just sits motionless for the entire 30-60+
        second wait, which reads as frozen even with the light/sound cues."""
        if self.dialogue_worker is None or not self.dialogue_worker.is_busy():
            return
        if time.time() < self._next_think_pulse_at:
            return
        self.executor.run(Action(kind="think", params={}))
        self._next_think_pulse_at = time.time() + THINK_PULSE_INTERVAL_S

    def _check_dialogue_reply(self, currently_engaged: bool) -> None:
        if self.dialogue_worker is None:
            return
        reply = self.dialogue_worker.get_ready_reply()
        if reply is None:
            return
        if not currently_engaged:
            # Disengaged while the reply was still in flight -- don't have
            # it speak into an empty room once it finally arrives.
            self._on_debug(f'(reply ready but no longer engaged, dropping: "{reply.reply}")')
            return
        self._restore_engaged_light()
        # Start the audio playing first (non-blocking -- see on_speak_audio),
        # *then* run the gesture, so the gesture happens while it's actually
        # talking instead of before or after.
        self._speak_synthesized(reply.audio_bytes)
        if reply.gesture != "none":
            self.executor.run(Action(kind=reply.gesture, params={}))

    def _speak_synthesized(self, audio_bytes: bytes) -> None:
        """Plays audio already synthesized by DialogueWorker -- deliberately
        bypasses the speak Action/on_speak hook, which would call Gemini's
        TTS *again* (another ~15s network call) for audio we already have."""
        try:
            self.executor.hooks.on_speak_audio(audio_bytes)
        except Exception as exc:  # noqa: BLE001
            self._on_debug(f"PLAYBACK FAILED: {exc}")

    def _restore_engaged_light(self) -> None:
        self.executor.run(
            Action(kind="set_light", params={"on": True, "color": WARM_WHITE, "brightness": ENGAGED_BRIGHTNESS})
        )

    def _report_failures(self) -> None:
        for t in self.executor.drain_telemetry():
            if t.kind == "action_failed":
                self._on_debug(f"ACTION FAILED: {t.payload}")

    def _on_engage(self, face_x_frac: float) -> None:
        pan = -face_x_frac * MAX_PAN_RAD
        self._on_debug(f"  -> look_at pan={pan:.2f} rad")
        # The idle music stops and a chime marks the moment of noticing --
        # music and "I'm listening to you now" shouldn't overlap once
        # we're actually paying attention to someone.
        self.executor.run(Action(kind="music_off", params={}))
        self.executor.run(Action(kind="play_sound", params={"name": "engage_chime.wav"}))
        # "Oh, I see you!" -- a quick bright/dim flash right as it notices,
        # before it even finishes turning, then settle into full brightness
        # once it's actually looking at and focused on the person.
        self._set_light(NOTICE_FLASH_COUNT * [ENGAGED_BRIGHTNESS, 0.05], interval=NOTICE_FLASH_INTERVAL_S)
        self.executor.run(Action(kind="look_at", params={"pan": pan, "tilt": -0.1}))
        self.executor.run(Action(kind="nod", params={}))
        self.executor.run(
            Action(kind="set_light", params={"on": True, "color": WARM_WHITE, "brightness": ENGAGED_BRIGHTNESS})
        )
        if self.speech is not None:
            self.speech.set_active(True)

    def _on_disengage(self) -> None:
        if self.speech is not None:
            self.speech.set_active(False)
        self.executor.run(
            Action(kind="set_light", params={"on": True, "color": WARM_WHITE, "brightness": IDLE_BRIGHTNESS})
        )
        self.executor.run(Action(kind="home", params={}))
        self.executor.run(Action(kind="music_on", params={}))
        # Give it a moment to settle at home before wandering starts again,
        # rather than immediately drifting off right as it returns.
        self._next_idle_wander_at = self._schedule_next_idle_wander()

    def _set_light(self, brightness_sequence: list[float], interval: float) -> None:
        for brightness in brightness_sequence:
            self.executor.run(
                Action(kind="set_light", params={"on": True, "color": WARM_WHITE, "brightness": brightness})
            )
            time.sleep(interval)
