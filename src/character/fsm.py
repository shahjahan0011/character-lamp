"""Ties perception signals to body actions.

Handles engagement (demo moments 1 and 2: notice someone, acknowledge them,
dim back down when they leave) and spoken interaction (demo moment 3: only
listens while engaged, transcribes, replies). Memory/goal states get added
here later without changing how this part works -- it just reads whatever
EngagementWatcher/SpeechCapture report and reacts.

Deliberately not calling the LLM for the engagement *reaction* (see design
discussion): it needs to feel instantaneous, and a network round-trip would
undercut that, so the acknowledgment is a fixed, hardcoded Action sequence.
The spoken reply, by contrast, has to go through the LLM -- there's no
faking "understood what you said and answered it".
"""

from __future__ import annotations

import random
import time
from typing import Callable, Optional

from src.body.executor import ActionExecutor
from src.perception.engagement import EngagementWatcher
from src.protocol.models import Action
from src.speech.capture import SpeechCapture
from src.speech import dialogue, stt

MAX_PAN_RAD = 0.7  # radians; matches base_yaw_joint's usable range for a look

WARM_WHITE = [1.0, 0.95, 0.76]
NOTICE_FLASH_COUNT = 2
NOTICE_FLASH_INTERVAL_S = 0.12
IDLE_BRIGHTNESS = 0.2
ENGAGED_BRIGHTNESS = 1.0
# Free-tier Gemini audio calls are genuinely slow (measured live: STT
# ~34s, TTS ~16s) -- a cool, dim, distinct color while processing so the
# 30-50s round trip reads as "thinking", not "frozen/broken".
THINKING_COLOR = [0.55, 0.7, 1.0]
THINKING_BRIGHTNESS = 0.45

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
        on_debug: Optional[Callable[[str], None]] = None,
    ):
        self.executor = executor
        self.watcher = watcher
        self.speech = speech
        self._last_engaged = False
        # Optional hook for tests/scripts to print what's happening -- the
        # executor swallows action exceptions into telemetry by design (one
        # bad action shouldn't kill the demo), which otherwise means a
        # failure looks identical to "nothing happened". This surfaces it.
        self._on_debug = on_debug or (lambda msg: None)
        self._next_idle_wander_at = self._schedule_next_idle_wander()
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
        tests (and later, a bigger orchestrator loop covering speech/goals)
        can call it directly instead of only via a blocking loop."""
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
        self._report_failures()

    def _check_speech(self) -> None:
        utterance = self.speech.get_pending_utterance()
        if utterance is None:
            return
        self._on_debug(f"heard {utterance.duration_s:.1f}s of speech, transcribing...")
        self.executor.run(
            Action(kind="set_light", params={"on": True, "color": THINKING_COLOR, "brightness": THINKING_BRIGHTNESS})
        )
        try:
            transcript = stt.transcribe(utterance.wav_bytes)
        except Exception as exc:  # noqa: BLE001 -- one bad STT call shouldn't kill the demo
            self._on_debug(f"STT FAILED: {exc}")
            self._restore_engaged_light()
            return
        if not transcript:
            self._on_debug("(transcript empty -- likely no speech in that clip)")
            self._restore_engaged_light()
            return
        self._on_debug(f'  transcript: "{transcript}"')
        try:
            result = dialogue.respond(transcript)
        except Exception as exc:  # noqa: BLE001
            self._on_debug(f"DIALOGUE FAILED: {exc}")
            self._restore_engaged_light()
            return
        self._on_debug(f'  reply: "{result.reply}" (gesture={result.gesture})')
        self._restore_engaged_light()
        if result.gesture != "none":
            self.executor.run(Action(kind=result.gesture, params={}))
        self.executor.run(Action(kind="speak", params={"text": result.reply}))

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
