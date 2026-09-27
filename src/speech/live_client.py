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
import queue
import threading
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Callable, Literal, Optional

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000  # confirmed live: Live API's audio/pcm output mime rate

# Measured live before committing to this pipeline: gemini-2.5-flash-
# native-audio-preview-09-2025 landed ~1.7-2.2s to first audio on a warm,
# already-open connection across repeated real turns (one cold-start
# outlier hit 12s on a *fresh* connection, which is exactly why this
# client connects once for the app's lifetime rather than per-turn). The
# "-latest" alias tracks whichever current stable native-audio model that
# preview graduates into, rather than a dated preview snapshot; it tested
# with equivalent latency (~1.8s) against the same account/key.
LIVE_MODEL = "gemini-2.5-flash-native-audio-latest"

GestureName = Literal["nod", "shake_head", "excited", "curious", "think"]
GESTURE_NAMES = frozenset({"nod", "shake_head", "excited", "curious", "think"})

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
    "- think: briefly considering something before answering\n"
)


def _gesture_tool_declaration() -> dict:
    return {
        "name": "perform_gesture",
        "description": "Perform one bounded physical reaction gesture on the lamp's body.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": sorted(GESTURE_NAMES),
                }
            },
            "required": ["name"],
        },
    }


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
    audio: Optional[bytes] = None
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_call_id: Optional[str] = None
    message: Optional[str] = None


class GeminiLiveClient:
    def __init__(
        self,
        model: str,
        api_key: str,
        on_debug: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._on_debug = on_debug or (lambda msg: None)

        self._events: "queue.Queue[LiveEvent]" = queue.Queue()
        self._mic_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=200)
        self._tool_results: "queue.Queue[tuple[str, str, dict]]" = queue.Queue()
        self._text_nudges: "queue.Queue[str]" = queue.Queue()
        self._commit_requests: "queue.Queue[None]" = queue.Queue()

        self._mic_gate_open = threading.Event()
        self._connected = threading.Event()
        self._stop = threading.Event()

        # A bounded rolling transcript, built from the input/output
        # transcription Gemini already sends us (config requests it, but
        # nothing previously read it) -- restores conversational context
        # after a reconnect, since a fresh Live session otherwise starts
        # with zero memory of anything said before the drop. Also stands
        # in for "give it context about our conversation": the persistent
        # session already carries context turn-to-turn on its own; this is
        # specifically for surviving the *reconnects* that were silently
        # wiping it.
        self._history: "deque[tuple[str, str]]" = deque(maxlen=8)
        self._last_context_nudge: Optional[str] = None

        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()

    # -- public, thread-safe interface (called from the main thread) --------

    def wait_until_connected(self, timeout_s: float = 15.0) -> bool:
        return self._connected.wait(timeout_s)

    def is_connected(self) -> bool:
        return self._connected.is_set()

    def is_mic_gate_open(self) -> bool:
        return self._mic_gate_open.is_set()

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
            try:
                self._mic_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._mic_queue.put_nowait(pcm16_bytes)
            except queue.Full:
                pass

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

    def close(self) -> None:
        self._stop.set()

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
            tools=[{"function_declarations": [_gesture_tool_declaration()]}],
            input_audio_transcription={},
            output_audio_transcription={},
        )
        delays = (0.5, 1.0, 2.0, 4.0)
        attempt = 0
        first_connect = True
        while not self._stop.is_set():
            try:
                async with client.aio.live.connect(model=self._model, config=config) as session:
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
                                self._events.put(LiveEvent(kind="audio_chunk", audio=inline.data))
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
