"""Continuous microphone streaming for the persistent Gemini Live session.

Replaces capture.py's SpeechCapture (record one full utterance client-side,
upload the whole WAV, wait for a separate understand+reply+TTS round trip)
now that conversation goes over one always-open Live session instead: audio
streams to Gemini continuously, in small chunks, while the mic gate is open.

Still runs its own lightweight local loudness-based start/end detection
rather than relying purely on Gemini's own server-side turn detection --
confirmed live in the previous pipeline that a plain amplitude floor
(rms >= MIN_SPEECH_RMS) cleanly separates real speech from this room's
noise floor (fan/electrical hum measured rms 4-21; real speech measured
rms 123-1700+). Explicit local endpointing (calling commit_utterance() the
moment local silence returns) is also more deterministic than only hoping
the remote service notices a pause on its own -- the same reasoning behind
the reference implementation this pipeline is adapted from.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import sounddevice as sd

from .live_client import INPUT_SAMPLE_RATE, GeminiLiveClient

CHUNK_MS = 100
CHUNK_SAMPLES = INPUT_SAMPLE_RATE * CHUNK_MS // 1000

MIN_SPEECH_RMS = 60  # see capture.py's identical constant/rationale
SILENCE_CHUNKS_TO_END = 6  # ~600ms of quiet after speech ends the utterance


class LiveMicStreamer:
    def __init__(
        self,
        live_client: GeminiLiveClient,
        on_debug: Callable[[str], None] | None = None,
    ):
        self._live = live_client
        self._on_debug = on_debug or (lambda msg: None)
        self._stream: sd.InputStream | None = None
        self._speaking = False
        self._quiet_run = 0

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=INPUT_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SAMPLES,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _callback(self, indata, frames, time_info, status) -> None:
        if not self._live.is_mic_gate_open():
            # Gate closed (disengaged, or Gemini mid-turn/mid-playback) --
            # don't feed audio and don't run local speech detection either,
            # so a stray commit_utterance() never fires for a window where
            # nothing was actually sent.
            if self._speaking:
                self._speaking = False
                self._quiet_run = 0
            return

        chunk = bytes(indata)
        self._live.feed_mic_audio(chunk)

        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0
        loud = rms >= MIN_SPEECH_RMS

        if loud:
            if not self._speaking:
                self._speaking = True
                self._on_debug(f"live mic: speech started rms={rms:.0f}")
                self._live.notify_local_speech_started()
            self._quiet_run = 0
        elif self._speaking:
            self._quiet_run += 1
            if self._quiet_run >= SILENCE_CHUNKS_TO_END:
                self._speaking = False
                self._quiet_run = 0
                self._on_debug("live mic: speech ended, committing utterance")
                self._live.commit_utterance()
