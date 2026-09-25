"""Runs the slow network calls (audio understanding + reply, then TTS
synthesis) on their own background thread, so the main orchestrator loop
never blocks on them.

Measured live against the real API: a single combined audio-understanding-
plus-reply call plus TTS synthesis can take 30-60+ seconds combined. Doing
that inline in CharacterOrchestrator.tick() -- as an earlier version of
this code did -- freezes the *entire* character for that whole window:
engagement transitions stop being checked (confirmed live: disengage
stopped working the instant speech started), idle wander stops, and any
utterances SpeechCapture finishes in the meantime just silently overwrite
each other in its single-slot pending-utterance buffer.

Mirrors the same "background thread only does the slow/IO-bound part,
main thread does anything that touches the simulator" split already used
for EngagementWatcher and SpeechCapture. Unlike those two, this thread's
job (network calls) has no simulator access at all -- it hands back a
fully-formed ReadyReply (text + already-synthesized audio bytes), and the
main thread is the only thing that ever dispatches an Action or plays
audio.
"""

from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Callable, Optional

from src.speech import dialogue, tts
from src.speech.capture import Utterance
from src.speech.dialogue import Gesture


@dataclass
class ReadyReply:
    transcript: str
    reply: str
    gesture: Gesture
    audio_bytes: bytes


class DialogueWorker:
    def __init__(self, on_debug: Optional[Callable[[str], None]] = None):
        self._on_debug = on_debug or (lambda msg: None)
        self._input: queue.Queue[Utterance] = queue.Queue()
        self._busy = threading.Event()

        self._result_lock = threading.Lock()
        self._ready_reply: Optional[ReadyReply] = None

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, utterance: Utterance) -> bool:
        """Returns False (and drops the utterance) if a reply is already in
        flight -- one conversational turn at a time, rather than queuing up
        a backlog of stale replies to deliver one after another later."""
        if self._busy.is_set():
            self._on_debug("dialogue worker: still processing the last utterance, ignoring this one")
            return False
        self._input.put(utterance)
        return True

    def is_busy(self) -> bool:
        return self._busy.is_set()

    def get_ready_reply(self) -> Optional[ReadyReply]:
        """Non-blocking poll from the main thread; returns and clears
        whatever finished, if anything."""
        with self._result_lock:
            reply, self._ready_reply = self._ready_reply, None
        return reply

    def _run(self) -> None:
        while True:
            utterance = self._input.get()
            self._busy.set()
            try:
                self._process(utterance)
            except Exception:  # noqa: BLE001 -- one bad turn shouldn't kill the worker
                self._on_debug(f"dialogue worker error:\n{traceback.format_exc()}")
            finally:
                self._busy.clear()

    def _process(self, utterance: Utterance) -> None:
        self._on_debug(f"heard {utterance.duration_s:.1f}s of speech, understanding + replying...")
        result = dialogue.respond_to_audio(utterance.wav_bytes)
        if not result.transcript:
            self._on_debug("(no speech recognized in that clip)")
            return
        self._on_debug(f'  transcript: "{result.transcript}"')
        self._on_debug(f'  reply: "{result.reply}" (gesture={result.gesture})')
        audio_bytes = tts.synthesize(result.reply)
        with self._result_lock:
            self._ready_reply = ReadyReply(
                transcript=result.transcript,
                reply=result.reply,
                gesture=result.gesture,
                audio_bytes=audio_bytes,
            )
