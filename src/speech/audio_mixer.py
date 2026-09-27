"""Single-output-stream audio mixer: speech, one-shot SFX, and a looped
music bed, mixed together with music ducked under speech.

Adapted from a reference implementation
(HCL_SweChallenge/src/lamp_character/audio/mixer.py) that solved exactly
the problem this project's old playback.py had: several independent
sd.play() calls compete for the audio device instead of composing, and
there was no single, reliable "is anything audible right now" signal for
gating the microphone (the old fsm.py instead guessed a mute duration from
a clip's known length, which only covered TTS replies, not SFX/music).
Ported (not async, to match this project's plain-threading style -- the
original's async methods only ever did synchronous file IO + a lock-guarded
mutation) rather than reused directly, since that version is built for an
asyncio runtime this project doesn't otherwise have.

Speech audio is fed in small streamed PCM chunks as they arrive from the
Gemini Live session (see live_client.py), not as one pre-synthesized WAV --
this is what actually gets the lamp's voice starting within ~2s instead of
waiting for an entire reply to finish generating first.
"""

from __future__ import annotations

import os
import threading
import wave
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd

SFX_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "sfx")
MUSIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "music")
LOUNGE_LOOP_PATH = os.path.join(MUSIC_DIR, "lounge_loop.wav")

DUCK_MUSIC_DB = -18.0
SPEECH_GAIN = 1.0
SFX_GAIN = 0.9
MUSIC_GAIN = 0.5


def _pcm16(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0


class AudioMixer:
    """Owns one sounddevice OutputStream and mixes speech/SFX/music in its
    callback. All public methods are safe to call from any thread."""

    def __init__(self, sample_rate: int = 24000, channels: int = 1, prime_s: float = 0.2):
        self.sample_rate = sample_rate
        self.channels = channels
        self._speech: deque[np.ndarray] = deque(maxlen=200)
        self._sfx: deque[np.ndarray] = deque(maxlen=12)
        self._music: np.ndarray | None = None
        self._music_cursor = 0
        self._music_loop = False
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        # Confirmed live: playing speech chunks as soon as even one arrives
        # produces crackling/glitchy ("farting") audio -- chunks are
        # delivered to enqueue_speech_pcm16() in bursts (fsm.py only drains
        # the Live event queue once per ~100ms tick), while the output
        # callback consumes them continuously in real time, so the queue
        # can run dry between bursts. Buffering a small cushion before
        # starting playback (re-armed each time the queue fully drains, so
        # it applies at the start of every new reply) absorbs that jitter
        # at the cost of a small, fixed, one-time delay per turn -- well
        # within the 5s latency budget.
        self._prime_samples = int(prime_s * sample_rate)
        self._priming = True
        # sounddevice/PortAudio already reports real xruns via the
        # callback's `status` argument -- never previously read. Counting
        # them (rather than guessing from symptoms) is how to actually
        # confirm or rule out "buffer underrun" as the cause of an audio
        # glitch, and to correlate it against gesture timing (see
        # pop_xrun_counts() and fsm.py's use of it).
        self._underflow_count = 0
        self._overflow_count = 0

    def start(self) -> None:
        if self._stream is not None:
            return
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            latency="high",
            callback=self._callback,
        )
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _take(self, q: deque[np.ndarray], count: int) -> np.ndarray:
        output = np.zeros(count, dtype=np.float32)
        cursor = 0
        while q and cursor < count:
            chunk = q[0]
            consumed = min(count - cursor, len(chunk))
            output[cursor : cursor + consumed] += chunk[:consumed]
            cursor += consumed
            if consumed == len(chunk):
                q.popleft()
            else:
                q[0] = chunk[consumed:]
        return output

    def _callback(self, outdata, frames, time_info, status) -> None:
        if status.output_underflow:
            self._underflow_count += 1
        if status.output_overflow:
            self._overflow_count += 1
        with self._lock:
            if self._priming:
                buffered = sum(len(c) for c in self._speech)
                if buffered >= self._prime_samples:
                    self._priming = False
            if self._priming:
                speech = np.zeros(frames, dtype=np.float32)
            else:
                speech = self._take(self._speech, frames)
                if not self._speech:
                    # Drained dry -- re-arm priming so the *next* reply
                    # also gets a fresh buffering cushion instead of
                    # immediately starting from empty.
                    self._priming = True
            sfx = self._take(self._sfx, frames)
            music = np.zeros(frames, dtype=np.float32)
            if self._music is not None and len(self._music):
                remaining = len(self._music) - self._music_cursor
                consumed = min(frames, remaining)
                music[:consumed] = self._music[self._music_cursor : self._music_cursor + consumed]
                self._music_cursor += consumed
                if self._music_cursor >= len(self._music):
                    if self._music_loop:
                        self._music_cursor = 0
                    else:
                        self._music = None
                        self._music_cursor = 0
            duck = 10 ** (DUCK_MUSIC_DB / 20) if np.any(speech) else 1.0
            mixed = speech * SPEECH_GAIN + sfx * SFX_GAIN + music * MUSIC_GAIN * duck
        outdata[:, 0] = np.clip(mixed, -1.0, 1.0)

    # -- inputs (thread-safe) -------------------------------------------------

    def enqueue_speech_pcm16(self, pcm16_bytes: bytes) -> None:
        """Feeds one streamed chunk of 16-bit PCM speech audio (e.g. a Live
        session's audio deltas) -- appended, not replacing, so chunks play
        back-to-back in arrival order."""
        with self._lock:
            if len(self._speech) == self._speech.maxlen:
                self._speech.popleft()
            self._speech.append(_pcm16(pcm16_bytes))

    def interrupt_speech(self) -> None:
        """Drops any queued-but-not-yet-played speech -- used when Gemini
        reports the user barged in (server_content.interrupted)."""
        with self._lock:
            self._speech.clear()

    def play_sfx(self, path: Path) -> None:
        if not path.is_file():
            return
        data = self._read_wav(path, self.sample_rate)
        with self._lock:
            self._sfx.append(data)

    def play_music_loop(self, path: Path) -> None:
        if not path.is_file():
            return
        data = self._read_wav(path, self.sample_rate)
        with self._lock:
            self._music = data
            self._music_cursor = 0
            self._music_loop = True

    def stop_music(self) -> None:
        with self._lock:
            self._music = None
            self._music_cursor = 0
            self._music_loop = False

    @property
    def speech_pending(self) -> bool:
        """Whether Gemini's speech is still waiting to be played out."""
        with self._lock:
            return bool(self._speech)

    def pop_xrun_counts(self) -> tuple[int, int]:
        """Non-blocking poll: real (underflow, overflow) counts from
        PortAudio since the last call, for correlating against gesture
        timing (see fsm.py) instead of guessing from symptoms alone."""
        underflow, self._underflow_count = self._underflow_count, 0
        overflow, self._overflow_count = self._overflow_count, 0
        return underflow, overflow

    @property
    def output_pending(self) -> bool:
        """Whether any character-generated audio is currently audible --
        used to gate the microphone (see live_capture.py) so the lamp can't
        record and react to its own voice coming back out of the speaker."""
        with self._lock:
            return bool(self._speech or self._sfx or self._music is not None)

    def make_hooks(self) -> dict:
        """Builds the on_play_sound/on_music_on/on_music_off callables for
        ExecutorHooks, backed by this mixer -- the same Action vocabulary
        (play_sound/music_on/music_off) dispatches exactly as before, just
        through one shared output stream (with ducking) instead of
        independent sd.play() calls that could step on each other."""

        def on_play_sound(name: str) -> None:
            path = Path(name) if os.path.isabs(name) else Path(SFX_DIR) / name
            self.play_sfx(path)

        return {
            "on_play_sound": on_play_sound,
            "on_music_on": lambda: self.play_music_loop(Path(LOUNGE_LOOP_PATH)),
            "on_music_off": self.stop_music,
        }

    @staticmethod
    def _read_wav(path: Path, target_sample_rate: int) -> np.ndarray:
        with wave.open(str(path), "rb") as source:
            raw = source.readframes(source.getnframes())
            data = _pcm16(raw)
            if source.getnchannels() > 1:
                data = data.reshape(-1, source.getnchannels()).mean(axis=1)
            source_rate = source.getframerate()
        if source_rate != target_sample_rate and len(data):
            # Existing assets (scripts/generate_audio_assets.py) are 44.1kHz;
            # the mixer runs at Gemini Live's 24kHz output rate to avoid
            # resampling the latency-sensitive speech stream instead. Plain
            # linear interpolation is adequate for a short chime/hum/loop --
            # not hi-fi, but no extra dependency (scipy) for it either.
            duration_s = len(data) / source_rate
            target_len = max(1, round(duration_s * target_sample_rate))
            source_x = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
            target_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
            data = np.interp(target_x, source_x, data).astype(np.float32)
        return data
