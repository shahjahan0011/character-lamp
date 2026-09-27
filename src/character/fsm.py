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
from collections.abc import Callable

from pydantic import ValidationError

from src.body.executor import ActionExecutor
from src.character.character_state import (
    CharacterState,
    CharacterStateMachine,
    InvalidTransitionError,
)
from src.character.goal_coordinator import GoalCoordinator
from src.character.memory import SceneMemory
from src.character.metrics import MetricsLog
from src.character.observation_registry import ObservationRegistry
from src.character.scene_observer import SceneObserver
from src.character.tool_gateway import ToolGateway
from src.perception.engagement import EngagementWatcher
from src.protocol.models import Action
from src.protocol.tools import PerformGestureArgs
from src.speech.audio_mixer import AudioMixer
from src.speech.error_speech import ErrorSpeech
from src.speech.live_client import GeminiLiveClient, LiveEvent

# Don't re-announce a connection problem on every retry within a fast
# reconnect loop (0.5s/1s/2s/4s backoff, see live_client.py) -- one
# "having trouble connecting" is a useful cue, four in ten seconds is not.
CONNECTION_ERROR_ANNOUNCE_COOLDOWN_S = 20.0

# Every ActionKind that goes through ActionExecutor._move_and_settle()'s
# blocking sleep-and-step loop -- used only to correlate audio xruns
# against real movement telemetry (see _report_failures).
MOVEMENT_ACTION_KINDS = frozenset(
    {"look_at", "point_at", "nod", "shake_head", "excited", "curious", "think", "home", "idle_sway"}
)

# Measured live: this project's OutputStream reports ~144ms of hardware
# buffer latency regardless of the `latency` setting passed to it -- a
# real gap between "our software queue emptied" and "the speaker is
# actually silent". Reopening the mic gate immediately on queue-empty
# risks picking up that trailing ~144ms of the lamp's own voice.
MIC_REOPEN_GRACE_S = 0.25

MAX_PAN_RAD = 0.7  # radians; matches base_yaw_joint's usable range for a look

WARM_WHITE = [1.0, 0.95, 0.76]
NOTICE_FLASH_COUNT = 2
NOTICE_FLASH_INTERVAL_S = 0.12
IDLE_BRIGHTNESS = 0.2
ENGAGED_BRIGHTNESS = 1.0
# A brief warm/golden glow the instant local loudness detection notices
# someone start talking -- well before the utterance is actually committed
# (~400ms after they stop) -- so the character visibly reacts to being
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

# How often to sample this process's own CPU%/RSS into the metrics log --
# frequent enough to characterize steady-state load, not so frequent that
# the sampling itself (a psutil syscall) is a meaningful part of that load.
RESOURCE_SAMPLE_INTERVAL_S = 5.0


class CharacterOrchestrator:
    def __init__(
        self,
        executor: ActionExecutor,
        watcher: EngagementWatcher,
        live_client: GeminiLiveClient | None = None,
        audio_mixer: AudioMixer | None = None,
        observer: SceneObserver | None = None,
        memory: SceneMemory | None = None,
        registry: ObservationRegistry | None = None,
        goals: GoalCoordinator | None = None,
        gateway: ToolGateway | None = None,
        error_speech: ErrorSpeech | None = None,
        metrics: MetricsLog | None = None,
        on_debug: Callable[[str], None] | None = None,
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
        self.metrics = metrics if metrics is not None else MetricsLog(path=None, on_debug=self._on_debug)
        self._last_resource_sample_at = 0.0
        self._turn_started_monotonic = 0.0
        # NOT `memory or SceneMemory()` -- SceneMemory defines __len__, so an
        # empty-but-real instance passed in (the normal case: nothing's been
        # observed yet) is falsy and `or` would silently swap in a different
        # object than the caller shared with us.
        self.memory = memory if memory is not None else SceneMemory()
        self.registry = registry if registry is not None else ObservationRegistry()
        self.goals = goals if goals is not None else GoalCoordinator(self.registry)
        self.observer = observer if observer is not None else SceneObserver(
            memory=self.memory, registry=self.registry, on_debug=self._on_debug
        )
        self.gateway = gateway if gateway is not None else ToolGateway(
            executor=self.executor,
            observer=self.observer,
            registry=self.registry,
            memory=self.memory,
            goals=self.goals,
            get_latest_frame=self.watcher.get_latest_frame,
            on_debug=self._on_debug,
        )
        self.state_machine = CharacterStateMachine(on_debug=self._on_debug)
        self._last_engaged = False
        self._next_idle_wander_at = self._schedule_next_idle_wander()
        self._next_think_pulse_at = 0.0
        self._turn_in_flight = False
        self._turn_deadline = 0.0
        self._awaiting_first_audio = False
        self._next_reobserve_at = 0.0
        self._output_drained_at: float | None = None
        # Starts disengaged, so the idle music starts playing immediately.
        self.executor.run(Action(kind="music_on", params={}))

    def _try_transition(self, target: CharacterState) -> None:
        """The state machine formalizes/observes the same decisions the
        existing flags below already make -- it doesn't yet replace them
        as the sole source of truth (see module docstring), so an
        unexpected edge here is logged, not fatal to the demo."""
        try:
            self.state_machine.transition(target)
        except InvalidTransitionError as exc:
            self._on_debug(f"state machine: {exc}")

    def _schedule_next_idle_wander(self) -> float:
        return time.monotonic() + random.uniform(IDLE_WANDER_MIN_INTERVAL_S, IDLE_WANDER_MAX_INTERVAL_S)

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
            self.metrics.event("engaged", face_x_frac=state.face_x_frac)
            self._on_engage(state.face_x_frac)
        elif not state.engaged and self._last_engaged:
            self._on_debug("DISENGAGE")
            self.metrics.event("disengaged")
            self._on_disengage()
        elif not state.engaged and time.monotonic() >= self._next_idle_wander_at:
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
        self._sample_resources_if_due()

    def _sample_resources_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_resource_sample_at < RESOURCE_SAMPLE_INTERVAL_S:
            return
        self._last_resource_sample_at = now
        self.metrics.sample_resources()

    # -- Gemini Live event handling -------------------------------------------

    def _drain_live_events(self, currently_engaged: bool) -> None:
        for event in self.live_client.poll_events():
            if event.kind == "connected":
                self._on_debug("live: connected")
            elif event.kind == "speech_started":
                self.metrics.event("speech_started")
                if currently_engaged and not self._turn_in_flight:
                    self.executor.run(
                        Action(
                            kind="set_light",
                            params={"on": True, "color": LISTENING_COLOR, "brightness": ENGAGED_BRIGHTNESS},
                        )
                    )
            elif event.kind == "speech_committed":
                self.metrics.event("speech_committed")
                self._on_turn_started()
            elif event.kind == "audio_chunk":
                self._on_audio_chunk(currently_engaged)
            elif event.kind == "tool_call":
                self.metrics.event("tool_call", tool_name=event.tool_name)
                self._on_tool_call(event)
            elif event.kind == "turn_complete":
                self._on_turn_complete(currently_engaged)
            elif event.kind == "interrupted":
                self.audio_mixer.interrupt_speech()
                self._on_debug("live: playback interrupted (barge-in)")
            elif event.kind == "error":
                self.metrics.event("live_error", message=event.message)
                self._on_debug(f"live: error: {event.message}")
                if currently_engaged and self.error_speech is not None:
                    now = time.monotonic()
                    if now - self._last_connection_error_announced_at >= CONNECTION_ERROR_ANNOUNCE_COOLDOWN_S:
                        self._last_connection_error_announced_at = now
                        self.error_speech.say("connection")

    def _on_turn_started(self) -> None:
        """Local silence detection just committed the user's utterance
        (see live_capture.py) -- Gemini is now formulating a reply."""
        self._turn_in_flight = True
        self._turn_started_monotonic = time.monotonic()
        self._turn_deadline = self._turn_started_monotonic + TURN_TIMEOUT_S
        self._awaiting_first_audio = True
        self._try_transition(CharacterState.THINKING)
        self.executor.run(
            Action(
                kind="set_light",
                params={"on": True, "color": THINKING_COLOR, "brightness": THINKING_BRIGHTNESS},
            )
        )
        self.executor.run(Action(kind="play_sound", params={"name": "thinking_hum.wav"}))
        self._next_think_pulse_at = time.monotonic() + THINK_PULSE_INTERVAL_S

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
            latency_s = time.monotonic() - self._turn_started_monotonic
            self.metrics.event("first_audio", latency_s=latency_s)
            self._on_debug(f"live: first audio in {latency_s:.2f}s")
            self._awaiting_first_audio = False
            self._turn_deadline = time.monotonic() + TURN_TIMEOUT_S  # still speaking; extend the watchdog
            self._try_transition(CharacterState.SPEAKING)
            self._restore_engaged_light()

    def _on_tool_call(self, event: LiveEvent) -> None:
        """Every tool call goes through ToolGateway's strict validation
        and local invariant enforcement (goal ordering, observation
        freshness) -- this method only handles the Live-session plumbing
        around that: submitting a result immediately for a synchronous
        tool, or leaving it pending for request_observation (None means
        "queued on SceneObserver's background thread"; see
        _check_new_observation, which completes it once that finishes)."""
        name = event.tool_name or ""
        args = event.tool_args or {}
        call_id = event.tool_call_id or ""
        if name in ("look_at_image_point", "set_light", "perform_gesture"):
            self._try_transition(CharacterState.ACTING)
        if name == "perform_gesture":
            # This tool is declared NON_BLOCKING. Validate first, then
            # acknowledge that the gesture has started before running the
            # synchronous animation. Gemini can begin speaking immediately,
            # while the Live receive thread continues feeding audio directly
            # to the mixer during the movement.
            try:
                PerformGestureArgs.model_validate(args)
            except ValidationError:
                result = self.gateway.execute(name, args, call_id)
                self.live_client.submit_tool_result(call_id, name, result or {"ok": False})
                return
            self.live_client.submit_tool_result(call_id, name, {"ok": True, "status": "started"})
            self.gateway.execute(name, args, call_id)
            return
        result = self.gateway.execute(name, args, call_id)
        if result is not None:
            self.live_client.submit_tool_result(call_id, name, result)

    def _on_turn_complete(self, currently_engaged: bool) -> None:
        self._turn_in_flight = False
        self._awaiting_first_audio = False
        if currently_engaged:
            self._try_transition(CharacterState.ENGAGED)
            self._restore_engaged_light()
        # else: nothing to do -- _on_disengage() already set the idle
        # light, and _on_audio_chunk() already suppressed any audio.

    def _check_turn_timeout(self, currently_engaged: bool) -> None:
        if not self._turn_in_flight or time.monotonic() < self._turn_deadline:
            return
        self._on_debug("live: turn timed out (no audio/turn_complete) -- releasing mic gate")
        self._turn_in_flight = False
        self._awaiting_first_audio = False
        if currently_engaged:
            self._try_transition(CharacterState.ERROR)
            self.executor.run(Action(kind="shake_head", params={}))
            self._restore_engaged_light()
            self._try_transition(CharacterState.ENGAGED)
            if self.error_speech is not None:
                self.error_speech.say("timeout")

    def _check_thinking_pulse(self) -> None:
        """A slow, repeated 'pondering' dip while a reply is in flight but
        no audio has arrived yet -- keeps the character looking alive
        during the (now much shorter, but nonzero) wait for first audio."""
        if not self._turn_in_flight or not self._awaiting_first_audio:
            return
        if time.monotonic() < self._next_think_pulse_at:
            return
        self.executor.run(Action(kind="think", params={}))
        self._next_think_pulse_at = time.monotonic() + THINK_PULSE_INTERVAL_S

    def _update_mic_gate(self, currently_engaged: bool) -> None:
        """Recomputed every tick from live state rather than a guessed mute
        duration: open only while engaged, no turn in flight, and nothing
        the mixer's *software queue* is currently holding (speech/SFX/
        music). That queue emptying isn't quite the same moment as the
        speaker actually falling silent, though -- measured this output
        stream's own reported hardware buffer latency at ~144ms, a real
        gap during which the mic could reopen and pick up the tail end of
        the lamp's own voice. MIC_REOPEN_GRACE_S adds a fixed delay after
        the queue empties before actually reopening, to cover that gap."""
        output_pending = self.audio_mixer.output_pending if self.audio_mixer is not None else False
        if output_pending:
            self._output_drained_at = None
        elif self._output_drained_at is None:
            self._output_drained_at = time.monotonic()
        drained_long_enough = (
            self._output_drained_at is not None
            and time.monotonic() - self._output_drained_at >= MIC_REOPEN_GRACE_S
        )
        should_be_open = currently_engaged and not self._turn_in_flight and drained_long_enough
        self.live_client.set_mic_gate(should_be_open)

    # -- scene memory + goal observations ---------------------------------------

    def _check_reobserve(self, currently_engaged: bool) -> None:
        """Re-scans the surroundings periodically while engaged, not just
        once at the moment of engagement, so the lamp can notice something
        changed rather than only ever describing whatever was in view the
        instant someone first looked at it."""
        if not currently_engaged:
            return
        if time.monotonic() < self._next_reobserve_at:
            return
        self._observe_scene()
        self._next_reobserve_at = time.monotonic() + REOBSERVE_INTERVAL_S

    def _check_new_observation(self) -> None:
        if self.observer.pop_failure():
            # Silent by design for the *ambient* case: nobody's watching a
            # "vision call failed" cue specifically, and the person can
            # just ask again later. A tool-driven request_observation is
            # different -- Gemini is actually BLOCKING on it, so that one
            # gets a real structured tool error below instead of being
            # left hanging.
            self._on_debug("scene observer: last observation failed")
            if self.observer.pop_rate_limited():
                self._next_reobserve_at = time.monotonic() + REOBSERVE_BACKOFF_S
                self._on_debug(
                    f"scene observer: rate-limited, backing off {REOBSERVE_BACKOFF_S:.0f}s"
                )
            failed_call = self.observer.pop_failed_tool_call()
            if failed_call is not None and self.live_client is not None:
                call_id, tool_name, message = failed_call
                self.live_client.submit_tool_result(call_id, tool_name, {"ok": False, "error": message})

        result = self.observer.pop_new_observation()
        if result is None:
            return

        if result.tool_call_id is not None:
            # A tool-driven observation (object_memory/goal_planning/
            # goal_verification) -- advance the goal workflow if this was
            # for the active goal, then complete the BLOCKING tool call
            # with the real, structured result (never invented).
            self.gateway.complete_observation_for_goal(result.observation.observation_id, result.observation.purpose)
            if result.observation.purpose == "goal_verification":
                self._try_transition(CharacterState.VERIFYING)
            if self.live_client is not None:
                payload = {
                    "ok": True,
                    "observation_id": result.observation.observation_id,
                    "goal_id": self.gateway._current_goal_id,
                    "objects": [
                        {
                            "label": o.label,
                            "color": o.color,
                            "attributes": o.attributes,
                            "image_x": o.image_x,
                            "image_y": o.image_y,
                            "confidence": o.confidence,
                        }
                        for o in result.observation.objects
                    ],
                }
                self.live_client.submit_tool_result(result.tool_call_id, result.tool_name or "request_observation", payload)
            return

        # Ambient observations update local SceneMemory only. An earlier
        # version also injected the memory as realtime text into Gemini;
        # realtime text starts a model turn, so it raced the person's first
        # utterance and caused long or missing replies. Recall is already a
        # real tool, so the model can fetch this same memory on demand.

    def _observe_scene(self) -> None:
        """Kicks off one ambient scene-description call (demo moment 4) --
        grabs whatever frame EngagementWatcher's camera thread most
        recently captured rather than opening a second camera handle
        (most webcams only allow one consumer). Fire-and-forget: the
        result lands in self.memory whenever SceneObserver gets to it,
        and is relayed into the live conversation via
        _check_new_observation() above. Tool-driven request_observation
        calls (object_memory/goal_planning/goal_verification) go through
        ToolGateway instead, which submits directly to the same
        SceneObserver with a tool_call_id attached."""
        frame = self.watcher.get_latest_frame()
        if frame is None:
            return
        self.observer.submit_observation(frame, purpose="scene")

    def _restore_engaged_light(self) -> None:
        self.executor.run(
            Action(kind="set_light", params={"on": True, "color": WARM_WHITE, "brightness": ENGAGED_BRIGHTNESS})
        )

    def _report_failures(self) -> None:
        recent_movement = None
        for t in self.executor.drain_telemetry():
            if t.kind == "action_failed":
                self._on_debug(f"ACTION FAILED: {t.payload}")
            elif t.kind == "action_done" and t.payload.get("kind") in MOVEMENT_ACTION_KINDS:
                recent_movement = t.payload
        if self.audio_mixer is not None:
            underflow, overflow = self.audio_mixer.pop_xrun_counts()
            if underflow or overflow:
                self.metrics.event(
                    "audio_xrun",
                    underflow=underflow,
                    overflow=overflow,
                    movement=recent_movement["kind"] if recent_movement is not None else None,
                )
                # Real PortAudio-reported xruns (not a guess) -- logging
                # whether one coincided with a movement action this same
                # tick is exactly the evidence needed to confirm or rule
                # out "audio glitches happen because of gestures" as
                # opposed to gestures and speech simply always co-occurring
                # by design (Gemini calls gestures mid-reply).
                if recent_movement is not None:
                    self._on_debug(
                        f"AUDIO XRUN: underflow={underflow} overflow={overflow} -- "
                        f"coincided with movement '{recent_movement['kind']}' "
                        f"({recent_movement['elapsed_s'] * 1000:.0f}ms)"
                    )
                else:
                    self._on_debug(
                        f"AUDIO XRUN: underflow={underflow} overflow={overflow} "
                        "(no movement action this tick)"
                    )

    def _on_engage(self, face_x_frac: float) -> None:
        self._try_transition(CharacterState.NOTICING)
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
        self._try_transition(CharacterState.ENGAGED)
        self._observe_scene()
        self._next_reobserve_at = time.monotonic() + REOBSERVE_INTERVAL_S

    def _on_disengage(self) -> None:
        self._try_transition(CharacterState.DISENGAGING)
        if self.live_client is not None:
            # Clear engaged/audio-output permission FIRST, before the
            # blocking home motion below -- otherwise late-arriving audio
            # from a reply still in flight could keep playing for the
            # whole (blocking, potentially ~1s+) duration of the home
            # gesture, since GeminiLiveClient's own audio-feed gating
            # (see live_client.py's set_engaged) wouldn't be updated until
            # after this method returns.
            self.live_client.set_engaged(False)
            self.live_client.set_mic_gate(False)
        self.executor.run(
            Action(kind="set_light", params={"on": True, "color": WARM_WHITE, "brightness": IDLE_BRIGHTNESS})
        )
        self.executor.run(Action(kind="home", params={}))
        self.executor.run(Action(kind="music_on", params={}))
        # Give it a moment to settle at home before wandering starts again,
        # rather than immediately drifting off right as it returns.
        self._next_idle_wander_at = self._schedule_next_idle_wander()
        self._try_transition(CharacterState.DORMANT)

    def _set_light(self, brightness_sequence: list[float], interval: float) -> None:
        for brightness in brightness_sequence:
            self.executor.run(
                Action(kind="set_light", params={"on": True, "color": WARM_WHITE, "brightness": brightness})
            )
            time.sleep(interval)
