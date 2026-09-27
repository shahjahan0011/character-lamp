"""Persistent Gemini Live session -- the new conversational pipeline.

Replaces the old per-utterance chain (record a full WAV -> one "understand
+ reply" request/response call -> one separate TTS request/response call)
with a single always-open, bidirectional streaming session. That old chain
was the actual latency bottleneck: two sequential full network round trips
per turn, plus a persistent free-tier TTS quota wall (see gemini_client.py
git history -- gemini-2.5-flash-preview-tts capped at 10 requests/day).

Measured live against the real API before committing to this rewrite:
- A fresh connection's first turn can occasionally take 10+ seconds (one
  cold-start outlier observed out of ~10 real calls).
- An already-open, warm connection's turns land consistently ~1.7-2.2s to
  first audio across repeated real turns.
That's why this client connects ONCE for the app's whole lifetime (see
GeminiLiveClient.__init__) rather than per-engagement or per-turn -- the
user should never pay the cold-start cost.

Threading model mirrors the rest of this project deliberately:
EngagementWatcher/SpeechCapture/DialogueWorker each own a background thread
and expose a thread-safe, polled interface; the main thread is the only
thing that ever touches PyBullet or dispatches an Action. This client owns
a background thread running its own asyncio event loop (the google-genai
Live API is async-only), and exposes the same polling shape
(poll_events(), mirroring DialogueWorker.get_ready_reply()) so
CharacterOrchestrator.tick() can drain it without itself becoming async.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import queue
import threading
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from src.protocol.models import (  # noqa: F401 -- re-exported for existing callers
    GESTURE_NAMES,
    GestureName,
)
from src.protocol.tools import TOOL_ARG_MODELS

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000  # confirmed live: Live API's audio/pcm output mime rate

# Configurable via GEMINI_LIVE_MODEL. Measured live before picking this
# default: gemini-3.8-live landed ~1.2s to first audio and correctly
# waited on a Behavior.BLOCKING tool call before speaking (both confirmed
# against the real API). gemini-2.5-flash-native-audio-latest (this
# project's earlier default) also still works -- measured ~1.7-2.2s to
# first audio on a warm connection -- and remains available via
# GEMINI_LIVE_MODEL as a tested, documented fallback if gemini-3.8-live
# is ever unavailable on a given account.
LIVE_MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.8-live")
LIVE_MODEL_FALLBACK = "gemini-2.5-flash-native-audio-latest"

PERSONA = (
    "You are the voice of a small friendly desk lamp character (like "
    "Pixar's Luxo Jr., but able to talk). Someone is talking to you live "
    "through your microphone. Reply warmly and conversationally in 1-2 "
    "short sentences -- this is spoken aloud, not read, so keep it natural "
    "to say out loud, with real personality and warmth, not flat or "
    "robotic.\n\n"
    "You can also physically react -- use this often, it's a big part of "
    "how you come across as alive rather than just a voice. Call "
    "perform_gesture with the single reaction that best matches your "
    "reply's tone almost every time you speak (skip it only when truly "
    "nothing fits):\n"
    "- nod: agreeing, affirming, confirming something\n"
    "- shake_head: disagreeing, saying no, correcting a mistaken assumption\n"
    "- excited: delighted, enthusiastic, celebrating something\n"
    "- curious: intrigued, asking a question back, puzzling over something\n"
    "- think: briefly considering something before answering\n\n"
    "You have local memory of objects you've noticed. Never invent or "
    "guess what you remember -- always call recall_memory when asked what "
    "you've seen, noticed, or remember, and answer only from what it "
    "returns. If asked to remember a specific object, call "
    "request_observation(purpose=\"object_memory\") to get a fresh look, "
    "then remember_object with that observation's id. Background "
    "observations happen automatically -- don't narrate or mention them "
    "unless the person actually asks about the scene.\n\n"
    "When asked to find or act on something in the scene (e.g. \"find the "
    "red mug and point at it\"), follow this exact sequence: call "
    "request_observation(purpose=\"goal_planning\") first, always, even if "
    "you already have an older observation -- never act on stale "
    "information. Use the returned object coordinates with "
    "look_at_image_point (and set_light if appropriate) to act. "
    "Coordinates are approximate 2-D image positions, not precise 3-D "
    "locations, so describe pointing/looking, not exact distances. After "
    "acting, call request_observation(purpose=\"goal_verification\") to "
    "check the result, then call finish_goal with that verification "
    "observation's id -- never claim you completed a goal without "
    "finish_goal succeeding; if it's rejected, adapt and try again rather "
    "than insisting you already succeeded. If a tool call fails or "
    "returns an error, acknowledge it plainly (e.g. \"I couldn't get a "
    "clear look\") rather than pretending it worked. Distinguish "
    "'I cannot get a clear view right now' from 'I looked and it is not "
    "there' -- only say the second if a fresh observation actually didn't "
    "show it."
)


def _gemini_schema(value):
    """Strips JSON-Schema constructs Gemini's function-declaration parser
    doesn't accept (Pydantic emits $defs/title/additionalProperties that
    a plain function-call schema has no use for), and translates a Literal
    field's `const` into `enum` the same way. Mirrors a pattern confirmed
    necessary in a reference Live implementation this pipeline draws on."""
    if isinstance(value, dict):
        unsupported = {"$defs", "additionalProperties", "title"}
        translated = {k: _gemini_schema(v) for k, v in value.items() if k not in unsupported}
        if "const" in translated:
            translated["enum"] = [translated.pop("const")]
        return translated
    if isinstance(value, list):
        return [_gemini_schema(v) for v in value]
    return value


TOOL_DESCRIPTIONS = {
    "request_observation": (
        "Capture and analyze a fresh frame from the lamp's camera. Always "
        "call this before acting on or describing the current scene -- "
        "never rely on an old observation for a new decision."
    ),
    "recall_memory": (
        "Search locally remembered objects by a short text query. Always "
        "use this instead of guessing when asked what you've seen, "
        "noticed, or remember."
    ),
    "remember_object": (
        "Store one specific object from a fresh observation into local "
        "memory, so it can be recalled later."
    ),
    "look_at_image_point": (
        "Point the lamp's head/arm toward a normalized 2-D position from "
        "a specific observation. Approximate image-space pointing, not "
        "precise 3-D localization."
    ),
    "set_light": "Adjust the lamp's own light brightness/mode.",
    "perform_gesture": "Perform one bounded physical reaction gesture on the lamp's body.",
    "finish_goal": (
        "Report a goal's outcome. Only succeeds if a fresh goal_planning "
        "observation, at least one action, and a fresh goal_verification "
        "observation (captured after that action) were all recorded first."
    ),
}


def _tool_declarations() -> list[dict]:
    declarations = []
    for name, model_cls in TOOL_ARG_MODELS.items():
        parameters = _gemini_schema(model_cls.model_json_schema())
        declarations.append(
            {
                "name": name,
                "description": TOOL_DESCRIPTIONS[name],
                "parameters": parameters,
                # Gemini 3.8 Live's function calling is async by default;
                # every one of these tools has a local invariant that must
                # be satisfied *before* the model continues (a goal action
                # ordered before its observation, or a hung turn waiting on
                # a call we never actually answer) -- BLOCKING makes the
                # model wait for the real result rather than guessing
                # ahead. Confirmed live against gemini-3.8-live: the model
                # correctly waited for a BLOCKING tool's response before
                # producing any audio.
                "behavior": "BLOCKING",
            }
        )
    return declarations


@dataclass
class LiveEvent:
    """One thing that happened on the Live session, queued for the main
    thread to react to via CharacterOrchestrator.tick() -- see poll_events()."""

    kind: Literal[
        "connected",
        "speech_started",
        "speech_committed",
        "audio_chunk",
        "tool_call",
        "turn_complete",
        "interrupted",
        "error",
    ]
    audio: bytes | None = None
    tool_name: str | None = None
    tool_args: dict | None = None
    tool_call_id: str | None = None
    message: str | None = None


class GeminiLiveClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        on_audio_chunk: Callable[[bytes], None] | None = None,
        on_debug: Callable[[str], None] | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        # Called directly from the async receive-loop thread the instant a
        # chunk arrives -- confirmed live to be the fix for audio glitches
        # ("farting") specifically during gestures: gesture playback blocks
        # the *main* thread for its whole duration (ActionExecutor's
        # trajectory loop sleeps and steps PyBullet synchronously), and
        # audio used to only reach AudioMixer via that same main thread
        # (CharacterOrchestrator.tick() polling poll_events() and calling
        # enqueue_speech_pcm16()). A blocked main thread meant no new audio
        # reached the mixer for the gesture's whole duration, starving the
        # real-time output callback -- exactly when a gesture (like the
        # reactive ones Gemini calls mid-reply) was most likely to be
        # playing. Feeding the mixer from here instead means it keeps
        # getting fed regardless of what the main thread is doing.
        self._on_audio_chunk = on_audio_chunk
        self._on_debug = on_debug or (lambda msg: None)

        self._events: queue.Queue[LiveEvent] = queue.Queue()
        self._mic_queue: queue.Queue[bytes] = queue.Queue(maxsize=200)
        self._tool_results: queue.Queue[tuple[str, str, dict]] = queue.Queue()
        self._text_nudges: queue.Queue[str] = queue.Queue()
        self._commit_requests: queue.Queue[None] = queue.Queue()

        self._mic_gate_open = threading.Event()
        self._connected = threading.Event()
        self._stop = threading.Event()
        # Separate from mic gating (which also closes mid-turn/mid-
        # playback while still engaged) -- this specifically answers "is
        # anyone currently engaged at all", so a reply that finishes
        # arriving after they've looked away doesn't get played into an
        # empty room.
        self._engaged = threading.Event()

        # A bounded rolling transcript, built from the input/output
        # transcription Gemini already sends us (config requests it, but
        # nothing previously read it) -- restores conversational context
        # after a reconnect, since a fresh Live session otherwise starts
        # with zero memory of anything said before the drop. Also stands
        # in for "give it context about our conversation": the persistent
        # session already carries context turn-to-turn on its own; this is
        # specifically for surviving the *reconnects* that were silently
        # wiping it.
        self._history: deque[tuple[str, str]] = deque(maxlen=8)
        self._last_context_nudge: str | None = None

        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()

    # -- public, thread-safe interface (called from the main thread) --------

    def wait_until_connected(self, timeout_s: float = 15.0) -> bool:
        return self._connected.wait(timeout_s)

    def is_connected(self) -> bool:
        return self._connected.is_set()

    def is_mic_gate_open(self) -> bool:
        return self._mic_gate_open.is_set()

    def set_engaged(self, engaged: bool) -> None:
        """Called once per tick from CharacterOrchestrator -- gates the
        direct-to-mixer audio feed (see __init__) so audio still only ever
        plays while someone's actually engaged, even though it no longer
        routes through fsm.py's own engagement check."""
        if engaged:
            self._engaged.set()
        else:
            self._engaged.clear()

    def set_mic_gate(self, open_: bool) -> None:
        if open_:
            self._mic_gate_open.set()
        else:
            self._mic_gate_open.clear()
            # Purge anything already queued but not yet sent -- otherwise
            # up to ~20s of buffered pre-disengage audio (_mic_queue's
            # maxsize=200 * 100ms chunks) could still trickle out to
            # Gemini well after the gate closed, which is exactly what
            # "it listens even when disengaged" turned out to be.
            while True:
                try:
                    self._mic_queue.get_nowait()
                except queue.Empty:
                    break

    def feed_mic_audio(self, pcm16_bytes: bytes) -> None:
        """Called from the mic capture thread (see live_capture.py) --
        enqueues raw PCM to forward to Gemini. No-ops while the mic gate is
        closed (disengaged, or Gemini currently mid-turn/mid-playback) so
        audio just never reaches the model rather than needing to be
        filtered on the receiving end."""
        if not self._mic_gate_open.is_set():
            return
        try:
            self._mic_queue.put_nowait(pcm16_bytes)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                self._mic_queue.get_nowait()
            with contextlib.suppress(queue.Full):
                self._mic_queue.put_nowait(pcm16_bytes)

    def notify_local_speech_started(self) -> None:
        """Called from the mic capture thread (see live_capture.py) the
        instant local loudness detection notices someone started talking --
        lets the orchestrator give immediate "I hear you" feedback well
        before the utterance is actually committed (which waits for local
        silence, ~600ms after they stop -- see live_capture.py's
        SILENCE_CHUNKS_TO_END)."""
        self._events.put(LiveEvent(kind="speech_started"))

    def commit_utterance(self) -> None:
        """Signals local silence-detected end-of-speech (see live_capture.py)
        -- explicitly tells Gemini's automatic VAD the utterance is
        finished, rather than only relying on it to notice a pause on its
        own. Mirrors a deliberate choice in the reference implementation
        this pipeline is adapted from."""
        self._commit_requests.put(None)

    def send_text_nudge(self, text: str) -> None:
        """Injects a text turn into the live conversation -- used for the
        post-engagement scene-memory context so follow-up questions like
        "what did you see" can be answered from it. Also remembered as the
        most recent context nudge, so a reconnect can re-send it (see
        _build_reconnect_recap) instead of the model losing scene context
        along with everything else a session drop wipes out."""
        self._last_context_nudge = text
        self._text_nudges.put(text)

    def submit_tool_result(self, call_id: str, name: str, result: dict) -> None:
        self._tool_results.put((call_id, name, result))

    def poll_events(self) -> list[LiveEvent]:
        """Non-blocking drain, called once per tick() from the main thread
        -- mirrors DialogueWorker.get_ready_reply()'s polling contract."""
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                break
        return events

    def close(self, timeout_s: float = 2.0) -> None:
        """Bounded shutdown: signals both loops to stop and joins the
        background thread, but doesn't wait past timeout_s -- _receive_loop
        can be blocked inside `async for response in session.receive()`
        waiting on the next server message, which won't notice _stop until
        the next message (or connection close) arrives. The thread is a
        daemon (see __init__), so the process can still exit cleanly even
        if this specific join times out."""
        self._stop.set()
        self._thread.join(timeout=timeout_s)

    # -- background thread: owns its own asyncio event loop ------------------

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._main())
        except Exception:  # noqa: BLE001 -- last-resort: this thread must not die silently
            self._on_debug(f"live client fatal error:\n{traceback.format_exc()}")
        finally:
            loop.close()

    async def _main(self) -> None:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._api_key)
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=PERSONA,
            tools=[{"function_declarations": _tool_declarations()}],
            input_audio_transcription={},
            output_audio_transcription={},
        )
        delays = (0.5, 1.0, 2.0, 4.0)
        attempt = 0
        first_connect = True
        active_model = self._model
        fallback_attempted = False
        while not self._stop.is_set():
            try:
                async with client.aio.live.connect(model=active_model, config=config) as session:
                    self._connected.set()
                    self._events.put(LiveEvent(kind="connected"))
                    self._on_debug("live session connected")
                    if not first_connect:
                        # A reconnect otherwise starts with zero memory of
                        # the conversation -- restore what we can from our
                        # own bounded transcript + the last scene-memory
                        # nudge (see _build_reconnect_recap), rather than
                        # the model acting like it's meeting the person for
                        # the first time again mid-conversation.
                        recap = self._build_reconnect_recap()
                        if recap:
                            self._text_nudges.put(recap)
                            self._on_debug("live: queued reconnect recap to restore context")
                    first_connect = False
                    attempt = 0
                    await asyncio.gather(
                        self._send_loop(session, types),
                        self._receive_loop(session),
                    )
            except Exception as exc:  # noqa: BLE001 -- the connection itself died; must reconnect
                self._connected.clear()
                self._events.put(LiveEvent(kind="error", message=str(exc)))
                self._on_debug(f"live session error ({type(exc).__name__}), reconnecting: {exc}")
                if self._stop.is_set():
                    return
                if (
                    not fallback_attempted
                    and attempt >= 1
                    and active_model != LIVE_MODEL_FALLBACK
                ):
                    # The configured/default model failed to connect twice
                    # in a row -- fall back to the other model confirmed
                    # live to work with this pipeline (see LIVE_MODEL's
                    # docstring) rather than retrying the same broken
                    # model indefinitely.
                    self._on_debug(
                        f"live: '{active_model}' failed repeatedly, falling back to "
                        f"'{LIVE_MODEL_FALLBACK}'"
                    )
                    active_model = LIVE_MODEL_FALLBACK
                    fallback_attempted = True
                await asyncio.sleep(delays[min(attempt, len(delays) - 1)])
                attempt += 1

    def _build_reconnect_recap(self) -> str:
        parts = []
        if self._history:
            lines = []
            for user_text, model_text in self._history:
                if user_text:
                    lines.append(f"USER: {user_text}")
                if model_text:
                    lines.append(f"LAMP: {model_text}")
            if lines:
                parts.append(
                    "[Your connection just silently reconnected -- this is "
                    "the same ongoing conversation, not a new one. Here is "
                    "what was said so far; continue naturally, don't "
                    "re-greet them:]\n" + "\n".join(lines)
                )
        if self._last_context_nudge:
            parts.append(self._last_context_nudge)
        return "\n\n".join(parts)

    async def _send_loop(self, session, types) -> None:
        while not self._stop.is_set():
            did_something = False
            try:
                text = self._text_nudges.get_nowait()
                did_something = True
                try:
                    await session.send_realtime_input(text=text)
                    self._on_debug(f"live: sent text nudge ({len(text)} chars)")
                except Exception as exc:  # noqa: BLE001 -- isolate: one bad send shouldn't kill the session
                    self._on_debug(f"live: text nudge send failed, continuing: {exc}")
            except queue.Empty:
                pass

            try:
                self._commit_requests.get_nowait()
                did_something = True
                try:
                    await session.send_realtime_input(audio_stream_end=True)
                    self._events.put(LiveEvent(kind="speech_committed"))
                except Exception as exc:  # noqa: BLE001
                    self._on_debug(f"live: commit-utterance send failed, continuing: {exc}")
            except queue.Empty:
                pass

            try:
                call_id, name, result = self._tool_results.get_nowait()
                did_something = True
                try:
                    await session.send_tool_response(
                        function_responses=[
                            types.FunctionResponse(id=call_id, name=name, response=result)
                        ]
                    )
                except Exception as exc:  # noqa: BLE001 -- e.g. a stale/expired call_id
                    self._on_debug(f"live: tool-result send failed, continuing: {exc}")
            except queue.Empty:
                pass

            try:
                chunk = self._mic_queue.get_nowait()
                did_something = True
                try:
                    await session.send_realtime_input(
                        audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={INPUT_SAMPLE_RATE}")
                    )
                except Exception as exc:  # noqa: BLE001
                    self._on_debug(f"live: audio chunk send failed, continuing: {exc}")
            except queue.Empty:
                pass

            if not did_something:
                await asyncio.sleep(0.01)

    async def _receive_loop(self, session) -> None:
        user_transcript_parts: list[str] = []
        model_transcript_parts: list[str] = []
        while not self._stop.is_set():
            async for response in session.receive():
                try:
                    content = getattr(response, "server_content", None)
                    if content and getattr(content, "interrupted", False):
                        self._events.put(LiveEvent(kind="interrupted"))
                    if content:
                        input_t = getattr(content, "input_transcription", None)
                        if input_t and getattr(input_t, "text", None):
                            user_transcript_parts.append(input_t.text)
                        output_t = getattr(content, "output_transcription", None)
                        if output_t and getattr(output_t, "text", None):
                            model_transcript_parts.append(output_t.text)
                    if content and getattr(content, "model_turn", None):
                        for part in content.model_turn.parts:
                            inline = getattr(part, "inline_data", None)
                            if inline and inline.data:
                                # Fed to the mixer directly, from this
                                # thread, right now -- not routed through
                                # poll_events()/tick(), which can be
                                # blocked for a gesture's whole duration
                                # (see __init__'s docstring). The event is
                                # still published (without the bytes) so
                                # fsm.py can react to "first audio of this
                                # turn arrived" for the lighting cue.
                                if self._on_audio_chunk is not None and self._engaged.is_set():
                                    self._on_audio_chunk(inline.data)
                                self._events.put(LiveEvent(kind="audio_chunk"))
                    tool_call = getattr(response, "tool_call", None)
                    if tool_call:
                        for call in tool_call.function_calls:
                            self._events.put(
                                LiveEvent(
                                    kind="tool_call",
                                    tool_name=call.name,
                                    tool_args=dict(call.args or {}),
                                    tool_call_id=call.id,
                                )
                            )
                    if content and getattr(content, "turn_complete", False):
                        user_text = "".join(user_transcript_parts).strip()
                        model_text = "".join(model_transcript_parts).strip()
                        if user_text or model_text:
                            self._history.append((user_text, model_text))
                        user_transcript_parts = []
                        model_transcript_parts = []
                        self._events.put(LiveEvent(kind="turn_complete"))
                except Exception:  # noqa: BLE001 -- isolate: a bug processing one response
                    # shouldn't tear down a session that's otherwise fine.
                    self._on_debug(f"live: error handling one response, continuing:\n{traceback.format_exc()}")
