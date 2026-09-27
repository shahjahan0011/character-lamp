"""Spoken "something went wrong" announcements.

Pre-synthesized once at startup (not on demand) and played directly
through the shared AudioMixer, independent of the Gemini Live session --
deliberately, since one of the situations this needs to cover *is* the
Live connection itself being the thing that's wrong. Synthesizing on
demand would mean the one moment we need this to work reliably (things
are already failing) is also the moment a fresh network call is least
likely to succeed quickly.

Uses the same tts.py path as the standalone "speak" Action (a separate,
lightly-used model/quota from the Live session and from vision.py), so a
handful of short phrases synthesized once costs very little.
"""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
import soundfile as sf

from . import tts
from .audio_mixer import AudioMixer

PHRASES: dict[str, str] = {
    "connection": "Sorry, I'm having a little trouble connecting right now. One moment.",
    "timeout": "Sorry, that took a bit too long on my end. Could you try that again?",
    "generic": "Oops, something went wrong on my end there. Sorry about that.",
}


class ErrorSpeech:
    """Pre-synthesizes PHRASES once (see preload()) and plays a cached one
    on demand (see say()) -- never makes a network call from say() itself."""

    def __init__(self, mixer: AudioMixer, on_debug: Callable[[str], None] | None = None):
        self._mixer = mixer
        self._on_debug = on_debug or (lambda msg: None)
        self._clips: dict[str, tuple[np.ndarray, int]] = {}

    def preload(self) -> None:
        for key, text in PHRASES.items():
            try:
                wav_bytes = tts.synthesize(text)
                data, samplerate = sf.read(io.BytesIO(wav_bytes), dtype="int16")
                self._clips[key] = (data, samplerate)
            except Exception as exc:  # noqa: BLE001 -- best-effort; missing key just stays silent
                self._on_debug(f"error_speech: failed to preload '{key}': {exc}")

    def say(self, key: str) -> bool:
        """Plays a pre-synthesized phrase through the shared mixer.
        Returns whether a clip was actually available to play (preload()
        may not have been called, or that specific phrase may have failed
        to synthesize -- e.g. if the network was already down at
        startup)."""
        clip = self._clips.get(key)
        if clip is None:
            return False
        data, samplerate = clip
        pcm16_bytes = self._resample_to_mixer_rate(data, samplerate).tobytes()
        self._mixer.enqueue_speech_pcm16(pcm16_bytes)
        return True

    def _resample_to_mixer_rate(self, data: np.ndarray, source_rate: int) -> np.ndarray:
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.int16)
        target_rate = self._mixer.sample_rate
        if source_rate == target_rate:
            return data
        duration_s = len(data) / source_rate
        target_len = max(1, round(duration_s * target_rate))
        source_x = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
        target_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        return np.interp(target_x, source_x, data.astype(np.float32)).astype(np.int16)
