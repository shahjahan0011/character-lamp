"""Ties perception signals to body actions.

Handles engagement (demo moments 1 and 2: notice someone, acknowledge them,
dim back down when they leave), spoken interaction (demo moment 3: a
persistent Gemini Live session, not a per-utterance request/response
chain -- see live_client.py for why), and scene memory (demo moment 4:
describe whatever's in view once per engagement, so a later spoken
question can be answered from it). Goal states get added here later
without changing how this part works -- it just reads whatever
EngagementWatcher/GeminiLiveClient/SceneObserver report and reacts.

Deliberately not calling the LLM for the engagement *reaction*: it needs to
feel instantaneous, and a network round-trip would undercut that, so the
acknowledgment is a fixed, hardcoded Action sequence. The spoken
conversation, by contrast, has to go through the model -- there's no
faking "understood what you said and answered it".

This module previously drove a per-utterance pipeline (record a full
utterance -> one understand+reply call -> one separate TTS call) via
DialogueWorker's own background thread. That's gone: it measured 30-90+
seconds round trip, hit a hard 10-requests/day free-tier TTS quota wall,
and needed elaborate mitigation (retries, a timer-based mic-mute guess) for
a problem a persistent Live session doesn't have in the first place. Now
GeminiLiveClient owns one always-open bidirectional session (connected
once at startup -- reconnecting per-turn is what caused the occasional
10+ second cold-start latency observed live), CharacterOrchestrator polls
its event queue every tick() the same way it already polled
EngagementWatcher/SceneObserver, and mic gating is a live, per-tick
computed boolean (engaged, no turn in flight, nothing currently audible)
rather than a guessed mute duration.
"""

from __future__ import annotations

import random
import time
from typing import Callable, Optional

from src.body.executor import ActionExecutor
from src.character.memory import SceneMemory
from src.character.scene_observer import SceneObserver
from src.perception.engagement import EngagementWatcher
from src.protocol.models import Action
from src.speech.audio_mixer import AudioMixer
from src.speech.error_speech import ErrorSpeech
from src.speech.live_client import GESTURE_NAMES, GeminiLiveClient, LiveEvent

# Don't re-announce a connection problem on every retry within a fast
# reconnect loop (0.5s/1s/2s/4s backoff, see live_client.py) -- one
# "having trouble connecting" is a useful cue, four in ten seconds is not.
CONNECTION_ERROR_ANNOUNCE_COOLDOWN_S = 20.0

MAX_PAN_RAD = 0.7  # radians; matches base_yaw_joint's usable range for a look

WARM_WHITE = [1.0, 0.95, 0.76]
NOTICE_FLASH_COUNT = 2
NOTICE_FLASH_INTERVAL_S = 0.12
IDLE_BRIGHTNESS = 0.2
ENGAGED_BRIGHTNESS = 1.0
# A brief warm/golden glow the instant local loudness detection notices
# someone start talking -- well before the utterance is actually committed
# (~600ms after they stop) -- so the character visibly reacts to being
# spoken to immediately, not just once it's done "thinking" about a reply.
LISTENING_COLOR = [1.0, 0.85, 0.5]
# A cool, dim, distinct color, a repeated gentle "pondering" gesture, and a
# soft hum right as a turn starts, so even the (now much shorter) wait for
# first audio reads as active thinking rather than a frozen character.
THINKING_COLOR = [0.55, 0.7, 1.0]
THINKING_BRIGHTNESS = 0.45
THINK_PULSE_INTERVAL_S = 3.5
# A turn that neither produces audio nor completes within this long is
# almost certainly stuck (dropped connection, model error) -- release the
# mic gate and surface it rather than leaving the character stuck
# "thinking" and unable to hear anything new indefinitely.
TURN_TIMEOUT_S = 20.0

# How often to re-look at the surroundings while engaged, not just once at
# the moment of engagement -- so the lamp can notice something changed
# (a new object appeared/moved) instead of only ever describing whatever
# was in view the instant someone first looked at it.
#
# Confirmed live at 15s: a vision-model free-tier quota of 20 requests/day
# gets exhausted within about 5 minutes of continuous engagement, and every
# failed retry afterward hammered the same already-exhausted quota. 45s
# keeps re-observation "constant"/"regular" for a demo while giving a
# 20/day cap roughly 15 engaged minutes before running out -- plenty for
# any one continuous demo session, not infinite for an all-day dev loop.
REOBSERVE_INTERVAL_S = 45.0
# Backing off this much longer after a rate/quota failure (see
# pop_rate_limited()) means one exhausted-quota moment produces one
# retry-and-back-off, not a new failed attempt (and traceback) every
# single normal interval until the quota resets.
REOBSERVE_BACKOFF_S = 300.0

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
        live_client: Optional[GeminiLiveClient] = None,
        audio_mixer: Optional[AudioMixer] = None,
        observer: Optional[SceneObserver] = None,
        memory: Optional[SceneMemory] = None,
        error_speech: Optional[ErrorSpeech] = None,
        on_debug: Optional[Callable[[str], None]] = None,
    ):
        self.executor = executor
        self.watcher = watcher
        self.live_client = live_client
        self.audio_mixer = audio_mixer
        self.error_speech = error_speech
        self._last_connection_error_announced_at = 0.0
        # Optional hook for tests/scripts to print what's happening -- the
        # executor swallows action exceptions into telemetry by design (one
        # bad action shouldn't kill the demo), which otherwise means a
        # failure looks identical to "nothing happened". This surfaces it.
        self._on_debug = on_debug or (lambda msg: None)
        # NOT `memory or SceneMemory()` -- SceneMemory defines __len__, so an
        # empty-but-real instance passed in (the normal case: nothing's been
        # observed yet) is falsy and `or` would silently swap in a different
        # object than the caller shared with us.
        self.memory = memory if memory is not None else SceneMemory()
        self.observer = observer if observer is not None else SceneObserver(
            memory=self.memory, on_debug=self._on_debug
        )
        self._last_engaged = False
        self._next_idle_wander_at = self._schedule_next_idle_wander()
        self._next_think_pulse_at = 0.0
        self._turn_in_flight = False
        self._turn_deadline = 0.0
        self._awaiting_first_audio = False
        self._next_reobserve_at = 0.0
        self._nudged_this_engagement = False
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
        self._last_engaged = state.engaged
        # Checked every tick regardless of the branch above -- Live events
        # and observation results can arrive at any moment, independent of
        # whatever else is happening.
        if self.live_client is not None:
            self.live_client.set_engaged(state.engaged)
            self._drain_live_events(state.engaged)
            self._check_turn_timeout(state.engaged)
            self._check_thinking_pulse()
            self._update_mic_gate(state.engaged)
        self._check_reobserve(state.engaged)
        self._check_new_observation()
        self._report_failures()

    # -- Gemini Live event handling -------------------------------------------

    def _drain_live_events(self, currently_engaged: bool) -> None:
        for event in self.live_client.poll_events():
            if event.kind == "connected":
                self._on_debug("live: connected")
            elif event.kind == "speech_started":
                if currently_engaged and not self._turn_in_flight:
                    self.executor.run(
                        Action(
                            kind="set_light",
                            params={"on": True, "color": LISTENING_COLOR, "brightness": ENGAGED_BRIGHTNESS},
                        )
                    )
            elif event.kind == "speech_committed":
                self._on_turn_started()
            elif event.kind == "audio_chunk":
                self._on_audio_chunk(currently_engaged)
            elif event.kind == "tool_call":
                self._on_tool_call(event)
            elif event.kind == "turn_complete":
                self._on_turn_complete(currently_engaged)
            elif event.kind == "interrupted":
                self.audio_mixer.interrupt_speech()
                self._on_debug("live: playback interrupted (barge-in)")
            elif event.kind == "error":
                self._on_debug(f"live: error: {event.message}")
                if currently_engaged and self.error_speech is not None:
                    now = time.time()
                    if now - self._last_connection_error_announced_at >= CONNECTION_ERROR_ANNOUNCE_COOLDOWN_S:
                        self._last_connection_error_announced_at = now
                        self.error_speech.say("connection")

    def _on_turn_started(self) -> None:
        """Local silence detection just committed the user's utterance
        (see live_capture.py) -- Gemini is now formulating a reply."""
        self._turn_in_flight = True
        self._turn_deadline = time.time() + TURN_TIMEOUT_S
        self._awaiting_first_audio = True
        self.executor.run(
            Action(
                kind="set_light",
                params={"on": True, "color": THINKING_COLOR, "brightness": THINKING_BRIGHTNESS},
            )
        )
        self.executor.run(Action(kind="play_sound", params={"name": "thinking_hum.wav"}))
        self._next_think_pulse_at = time.time() + THINK_PULSE_INTERVAL_S

    def _on_audio_chunk(self, currently_engaged: bool) -> None:
        """The actual audio bytes are fed to the mixer directly from
        GeminiLiveClient's own receive thread (see live_client.py's
        __init__ docstring for why -- this used to route through here,
        which meant a blocked main thread, e.g. mid-gesture, starved the
        mixer of new audio). This handler only reacts to the *event* for
        the "first audio of this turn arrived" lighting cue; the
        engagement check that gates whether audio actually plays lives in
        GeminiLiveClient.set_engaged(), called every tick below."""
        if not currently_engaged:
            return
        if self._awaiting_first_audio:
            self._awaiting_first_audio = False
            self._turn_deadline = time.time() + TURN_TIMEOUT_S  # still speaking; extend the watchdog
            self._restore_engaged_light()

    def _on_tool_call(self, event: LiveEvent) -> None:
        name = (event.tool_args or {}).get("name")
        valid = name in GESTURE_NAMES
        if valid:
            self.executor.run(Action(kind=name, params={}))
        else:
            self._on_debug(f"live: rejected invalid gesture tool call: {event.tool_args!r}")
        # Gemini's turn stays open waiting for this -- an unknown/invalid
        # name still gets an explicit response so the turn isn't left
        # hanging, it just reports ok=False rather than performing anything.
        self.live_client.submit_tool_result(
            event.tool_call_id or "", event.tool_name or "perform_gesture", {"ok": valid}
        )

    def _on_turn_complete(self, currently_engaged: bool) -> None:
        self._turn_in_flight = False
        self._awaiting_first_audio = False
        if currently_engaged:
            self._restore_engaged_light()
        # else: nothing to do -- _on_disengage() already set the idle
        # light, and _on_audio_chunk() already suppressed any audio.

    def _check_turn_timeout(self, currently_engaged: bool) -> None:
        if not self._turn_in_flight or time.time() < self._turn_deadline:
            return
        self._on_debug("live: turn timed out (no audio/turn_complete) -- releasing mic gate")
        self._turn_in_flight = False
        self._awaiting_first_audio = False
        if currently_engaged:
            self.executor.run(Action(kind="shake_head", params={}))
            self._restore_engaged_light()
            if self.error_speech is not None:
                self.error_speech.say("timeout")

    def _check_thinking_pulse(self) -> None:
        """A slow, repeated 'pondering' dip while a reply is in flight but
        no audio has arrived yet -- keeps the character looking alive
        during the (now much shorter, but nonzero) wait for first audio."""
        if not self._turn_in_flight or not self._awaiting_first_audio:
            return
        if time.time() < self._next_think_pulse_at:
            return
        self.executor.run(Action(kind="think", params={}))
        self._next_think_pulse_at = time.time() + THINK_PULSE_INTERVAL_S

    def _update_mic_gate(self, currently_engaged: bool) -> None:
        """Recomputed every tick from live state rather than a guessed mute
        duration: open only while engaged, no turn in flight, and nothing
        the mixer is currently playing (speech/SFX/music) could be picked
        back up by the mic."""
        output_pending = self.audio_mixer.output_pending if self.audio_mixer is not None else False
        should_be_open = currently_engaged and not self._turn_in_flight and not output_pending
        self.live_client.set_mic_gate(should_be_open)

    # -- scene memory ----------------------------------------------------------

    def _check_reobserve(self, currently_engaged: bool) -> None:
        """Re-scans the surroundings periodically while engaged, not just
        once at the moment of engagement, so the lamp can notice something
        changed rather than only ever describing whatever was in view the
        instant someone first looked at it."""
        if not currently_engaged:
            return
        if time.time() < self._next_reobserve_at:
            return
        self._observe_scene()
        self._next_reobserve_at = time.time() + REOBSERVE_INTERVAL_S

    def _check_new_observation(self) -> None:
        if self.observer.pop_failure():
            # Silent by design: nobody's watching a "vision call failed"
            # cue specifically, and the person can just ask again later --
            # unlike a failed spoken turn, there's no waiting conversational
            # turn to visibly resolve.
            self._on_debug("scene observer: last observation failed")
            if self.observer.pop_rate_limited():
                self._next_reobserve_at = time.time() + REOBSERVE_BACKOFF_S
                self._on_debug(
                    f"scene observer: rate-limited, backing off {REOBSERVE_BACKOFF_S:.0f}s"
                )
        result = self.observer.pop_new_observation()
        if result is None:
            return
        objects, changed = result
        # Confirmed live: injecting scene-memory text mid-conversation via
        # send_realtime_input(text=...) makes Gemini speak in response
        # EVERY time, even with an explicit "don't say anything, this is
        # silent" instruction in the text itself -- that's what "randomly
        # started describing the scenery" turned out to be. The Live API
        # also explicitly warns against mixing send_client_content (whose
        # turn_complete=False *would* inject silently) with the continuous
        # send_realtime_input audio streaming this pipeline already
        # depends on. Given that, only relay a nudge once per engagement
        # (right after the first observation completes) rather than on
        # every periodic re-observation -- local memory still updates
        # continuously either way (see _check_reobserve), so a later
        # question is answered from whatever's freshest in self.memory;
        # it just isn't volunteered mid-conversation on its own.
        if self._nudged_this_engagement:
            return
        self._nudged_this_engagement = True
        if self.live_client is not None:
            self.live_client.send_text_nudge(self.memory.as_context_text())

    def _observe_scene(self) -> None:
        """Kicks off one scene-description call (demo moment 4) -- grabs
        whatever frame EngagementWatcher's camera thread most recently
        captured rather than opening a second camera handle (most webcams
        only allow one consumer). Fire-and-forget: the result lands in
        self.memory whenever SceneObserver gets to it, and is relayed into
        the live conversation via _check_new_observation() above."""
        frame = self.watcher.get_latest_frame()
        if frame is None:
            return
        self.observer.submit_observation(frame)

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
        # Mic gate opens on the next tick via _update_mic_gate (engaged,
        # no turn in flight, nothing playing) -- no separate call needed.
        self._nudged_this_engagement = False
        self._observe_scene()
        self._next_reobserve_at = time.time() + REOBSERVE_INTERVAL_S

    def _on_disengage(self) -> None:
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
