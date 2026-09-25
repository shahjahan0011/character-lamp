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
import traceback
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import webrtcvad

SAMPLE_RATE = 16000  # webrtcvad only accepts 8000/16000/32000/48000
FRAME_MS = 30  # webrtcvad only accepts 10/20/30ms frames
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

# How long to keep listening after engagement drops if we're mid-utterance
# when it does. Without this, a person naturally glancing down/away while
# still mid-sentence (extremely common -- you don't stare at a webcam while
# forming a thought) gets disengaged by the face detector, which used to
# discard the in-progress recording immediately, before the silence-based
# endpointing below ever got a chance to notice they'd actually finished
# talking. Confirmed via a live test: "speech started" fired every time,
# "utterance finished" never did -- DISENGAGE always cut in first.
GRACE_PERIOD_S = 1.5


def _pcm16_to_wav_bytes(pcm_bytes: bytes) -> bytes:
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    # Observed live: laptop-mic recordings peaking around rms 4-9 (vs.
    # ~1000+ typical for normal speaking volume into a real mic) -- a WAV
    # that quiet transcribes poorly regardless of how well it's segmented.
    # Normalize toward a healthy peak, capping the gain so near-total
    # silence (no real signal at all) doesn't just get amplified into loud
    # noise.
    peak = np.abs(audio).max()
    if peak > 0:
        gain = min(32767 * 0.9 / peak, 20.0)
        audio = audio * gain
    audio = np.clip(audio, -32768, 32767).astype(np.int16)
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
        on_debug: Optional[Callable[[str], None]] = None,
    ):
        # A crash in the background capture thread would otherwise be
        # silent (daemon thread, nothing polls its liveness) -- exactly
        # "it can't hear me" with no visible cause. This surfaces both
        # thread errors and a periodic mic-level readout so it's possible
        # to tell "no audio is reaching us" apart from "audio is arriving
        # but the VAD isn't calling it speech" apart from "the thread died".
        self._on_debug = on_debug or (lambda msg: None)
        self._vad = webrtcvad.Vad(vad_aggressiveness)
        self._ring_size = max(1, padding_ms // FRAME_MS)
        self._silence_frames_to_end = max(1, silence_ms_to_end // FRAME_MS)
        self._min_speech_frames = max(1, min_utterance_ms // FRAME_MS)

        self._active = threading.Event()
        self._deactivated_at: float | None = None
        self._running = False
        self._stream: sd.InputStream | None = None
        self._thread: threading.Thread | None = None

        self._capture_lock = threading.Lock()
        self._pending_frames: list[bytes] = []

        self._utterance_lock = threading.Lock()
        self._pending_utterance: Utterance | None = None

    # -- control (called from the main/orchestrator thread) -----------------

    def set_active(self, active: bool) -> None:
        """Turning off doesn't instantly stop listening -- see
        GRACE_PERIOD_S. If nothing was mid-utterance, the grace period
        costs nothing (the run loop just sees silence and never triggers)."""
        if active:
            self._active.set()
            self._deactivated_at = None
        else:
            self._active.clear()
            self._deactivated_at = time.time()

    def _in_grace_period(self) -> bool:
        return self._deactivated_at is not None and (time.time() - self._deactivated_at) < GRACE_PERIOD_S

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
        if not self._active.is_set() and not self._in_grace_period():
            return
        with self._capture_lock:
            self._pending_frames.append(bytes(indata))

    def _run(self) -> None:
        # Pre-trigger ring buffer only: a short rolling window used solely to
        # detect when speech *starts* (and to grab a bit of pre-roll audio
        # once it does). It must stay small (padding_ms). Detecting when
        # speech *ends* needs a separate, unbounded consecutive-frame
        # counter -- reusing this same small buffer for that (an earlier
        # version of this code did) makes "unvoiced_count >= silence_frames_
        # to_end" impossible to ever satisfy whenever silence_ms_to_end is
        # longer than padding_ms, since the buffer can never hold more than
        # its maxlen. Confirmed via a live test: full seconds of measured
        # silence (0/67 frames flagged as speech) still never ended the
        # utterance.
        ring_buffer: collections.deque = collections.deque(maxlen=self._ring_size)
        triggered = False
        voiced_frames: list[bytes] = []
        consecutive_unvoiced = 0
        last_level_log = 0.0
        frames_since_log = 0
        voiced_since_log = 0

        while self._running:
            if not self._active.is_set() and not self._in_grace_period():
                if triggered:
                    self._on_debug("speech capture: grace period expired mid-utterance, discarding")
                triggered = False
                voiced_frames = []
                consecutive_unvoiced = 0
                ring_buffer.clear()
                with self._capture_lock:
                    self._pending_frames = []
                time.sleep(0.05)
                continue

            with self._capture_lock:
                frames, self._pending_frames = self._pending_frames, []
            if not frames:
                time.sleep(0.01)
                continue

            try:
                for frame in frames:
                    if len(frame) != FRAME_SAMPLES * 2:  # 2 bytes/sample (int16)
                        # webrtcvad requires an exact 10/20/30ms frame; a
                        # malformed one would otherwise raise and silently
                        # kill this whole thread (nothing else polls it).
                        self._on_debug(
                            f"speech capture: dropping malformed frame "
                            f"({len(frame)} bytes, expected {FRAME_SAMPLES * 2})"
                        )
                        continue

                    is_speech = self._vad.is_speech(frame, SAMPLE_RATE)
                    frames_since_log += 1
                    voiced_since_log += int(is_speech)
                    rms = np.sqrt(np.mean(np.frombuffer(frame, dtype=np.int16).astype(np.float32) ** 2))
                    now = time.time()
                    if now - last_level_log > 2.0:
                        self._on_debug(
                            f"speech capture: mic level rms={rms:.0f} "
                            f"({voiced_since_log}/{frames_since_log} frames flagged as speech)"
                        )
                        last_level_log = now
                        frames_since_log = 0
                        voiced_since_log = 0

                    if not triggered:
                        ring_buffer.append((frame, is_speech))
                        voiced_count = sum(1 for _, s in ring_buffer if s)
                        if voiced_count > 0.5 * ring_buffer.maxlen:
                            triggered = True
                            voiced_frames = [f for f, _ in ring_buffer]
                            ring_buffer.clear()
                            consecutive_unvoiced = 0
                            self._on_debug("speech capture: speech started")
                    else:
                        voiced_frames.append(frame)
                        consecutive_unvoiced = 0 if is_speech else consecutive_unvoiced + 1
                        if consecutive_unvoiced >= self._silence_frames_to_end:
                            # Trim the trailing silence itself off before
                            # finishing -- it's not part of what was said.
                            spoken_frames = voiced_frames[: -self._silence_frames_to_end]
                            if len(spoken_frames) >= self._min_speech_frames:
                                self._finish_utterance(spoken_frames)
                            else:
                                self._on_debug("speech capture: too short, discarding")
                            triggered = False
                            voiced_frames = []
                            consecutive_unvoiced = 0
                            ring_buffer.clear()
            except Exception:  # noqa: BLE001 -- keep the capture thread alive
                self._on_debug(f"speech capture thread error:\n{traceback.format_exc()}")
                triggered = False
                voiced_frames = []
                consecutive_unvoiced = 0
                ring_buffer.clear()

    def _finish_utterance(self, voiced_frames: list[bytes]) -> None:
        pcm_bytes = b"".join(voiced_frames)
        wav_bytes = _pcm16_to_wav_bytes(pcm_bytes)
        duration_s = len(voiced_frames) * FRAME_MS / 1000.0
        self._on_debug(f"speech capture: utterance finished ({duration_s:.1f}s)")
        with self._utterance_lock:
            self._pending_utterance = Utterance(wav_bytes=wav_bytes, duration_s=duration_s)
