"""Microphone capture + VAD-based utterance segmentation.

Only records while explicitly told to (see set_active) -- the character
shouldn't be listening when nobody's engaged with it, both as a matter of
being a good design (it's supposed to be an *aware* character, not an
always-on recorder) and because it avoids paying for/sending audio nobody
asked it to process.

Mirrors EngagementWatcher's shape deliberately: a background thread does
capture and segmentation only, and exposes the latest completed utterance
through a polled, lock-protected getter (get_pending_utterance) rather than
firing a callback directly from that thread. Everything that touches the
simulator or makes a network call (transcription, dialogue, TTS) happens on
the main thread that polls it, in CharacterOrchestrator.tick() -- the same
"perception thread only perceives" boundary used for the camera.
"""

from __future__ import annotations

import collections
import io
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad

SAMPLE_RATE = 16000  # webrtcvad only accepts 8000/16000/32000/48000
FRAME_MS = 30  # webrtcvad only accepts 10/20/30ms frames
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


def _pcm16_to_wav_bytes(pcm_bytes: bytes) -> bytes:
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@dataclass
class Utterance:
    wav_bytes: bytes
    duration_s: float


class SpeechCapture:
    def __init__(
        self,
        vad_aggressiveness: int = 2,
        silence_ms_to_end: int = 700,
        min_utterance_ms: int = 300,
        padding_ms: int = 300,
    ):
        self._vad = webrtcvad.Vad(vad_aggressiveness)
        self._ring_size = max(1, padding_ms // FRAME_MS)
        self._silence_frames_to_end = max(1, silence_ms_to_end // FRAME_MS)
        self._min_speech_frames = max(1, min_utterance_ms // FRAME_MS)

        self._active = threading.Event()
        self._running = False
        self._stream: sd.InputStream | None = None
        self._thread: threading.Thread | None = None

        self._capture_lock = threading.Lock()
        self._pending_frames: list[bytes] = []

        self._utterance_lock = threading.Lock()
        self._pending_utterance: Utterance | None = None

    # -- control (called from the main/orchestrator thread) -----------------

    def set_active(self, active: bool) -> None:
        """Engagement dropping mid-utterance discards whatever was buffered
        so far, rather than trying to salvage a half-finished recording --
        the person plausibly just glanced away, but we shouldn't guess."""
        if active:
            self._active.set()
        else:
            self._active.clear()
            with self._capture_lock:
                self._pending_frames = []

    def get_pending_utterance(self) -> Utterance | None:
        """Returns and clears the most recently completed utterance, if any
        (non-blocking) -- polled from CharacterOrchestrator.tick()."""
        with self._utterance_lock:
            utterance, self._pending_utterance = self._pending_utterance, None
        return utterance

    def start(self) -> None:
        self._running = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=FRAME_SAMPLES,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    # -- capture thread + its own audio-driver callback ----------------------

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if not self._active.is_set():
            return
        with self._capture_lock:
            self._pending_frames.append(bytes(indata))

    def _run(self) -> None:
        ring_buffer: collections.deque = collections.deque(maxlen=self._ring_size)
        triggered = False
        voiced_frames: list[bytes] = []

        while self._running:
            if not self._active.is_set():
                triggered = False
                voiced_frames = []
                ring_buffer.clear()
                time.sleep(0.05)
                continue

            with self._capture_lock:
                frames, self._pending_frames = self._pending_frames, []
            if not frames:
                time.sleep(0.01)
                continue

            for frame in frames:
                is_speech = self._vad.is_speech(frame, SAMPLE_RATE)
                if not triggered:
                    ring_buffer.append((frame, is_speech))
                    voiced_count = sum(1 for _, s in ring_buffer if s)
                    if voiced_count > 0.5 * ring_buffer.maxlen:
                        triggered = True
                        voiced_frames = [f for f, _ in ring_buffer]
                        ring_buffer.clear()
                else:
                    voiced_frames.append(frame)
                    ring_buffer.append((frame, is_speech))
                    unvoiced_count = sum(1 for _, s in ring_buffer if not s)
                    if unvoiced_count >= self._silence_frames_to_end:
                        if len(voiced_frames) >= self._min_speech_frames:
                            self._finish_utterance(voiced_frames)
                        triggered = False
                        voiced_frames = []
                        ring_buffer.clear()

    def _finish_utterance(self, voiced_frames: list[bytes]) -> None:
        pcm_bytes = b"".join(voiced_frames)
        wav_bytes = _pcm16_to_wav_bytes(pcm_bytes)
        duration_s = len(voiced_frames) * FRAME_MS / 1000.0
        with self._utterance_lock:
            self._pending_utterance = Utterance(wav_bytes=wav_bytes, duration_s=duration_s)
