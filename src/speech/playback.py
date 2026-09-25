"""Local audio playback: a looping background track (idle music) and
one-shot sound effects (the engagement chime), both via sounddevice.

Kept deliberately simple -- no mixing engine, no ducking. The music loop
and an SFX can play concurrently (sounddevice mixes multiple streams at the
OS level), which is exactly what we want for "chime plays over the tail of
the music as it fades," but there's no volume-ducking logic here; if that
turns out to sound bad in practice, that's the next thing to add, not
something to build speculatively now.
"""

from __future__ import annotations

import os
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf


class MusicLoop:
    """Plays a WAV file on repeat until stopped, in a background thread."""

    def __init__(self, path: str, volume: float = 0.5):
        self._data, self._samplerate = sf.read(path, dtype="float32")
        self._volume = volume
        self._stream: sd.OutputStream | None = None
        self._position = 0
        self._lock = threading.Lock()

    def _callback(self, outdata, frames, time_info, status) -> None:
        with self._lock:
            n = len(self._data)
            end = self._position + frames
            if end <= n:
                chunk = self._data[self._position : end]
            else:
                # Wrap around for a seamless loop.
                chunk = np.concatenate([self._data[self._position :], self._data[: end - n]])
            self._position = end % n
        if chunk.ndim == 1:
            chunk = chunk.reshape(-1, 1)
        outdata[:] = chunk * self._volume

    def start(self) -> None:
        if self._stream is not None:
            return
        channels = 1 if self._data.ndim == 1 else self._data.shape[1]
        self._stream = sd.OutputStream(
            samplerate=self._samplerate,
            channels=channels,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            with self._lock:
                self._position = 0

    def is_playing(self) -> bool:
        return self._stream is not None


def play_once(path: str, volume: float = 1.0, blocking: bool = False) -> None:
    """Fire-and-forget playback of a short sound effect."""
    data, samplerate = sf.read(path, dtype="float32")
    sd.play(data * volume, samplerate)
    if blocking:
        sd.wait()


SFX_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "sfx")
MUSIC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "music")

ENGAGE_CHIME_PATH = os.path.join(SFX_DIR, "engage_chime.wav")
LOUNGE_LOOP_PATH = os.path.join(MUSIC_DIR, "lounge_loop.wav")


def make_audio_hooks(music_volume: float = 0.5) -> dict:
    """Builds the on_play_sound/on_music_on/on_music_off callables for
    ExecutorHooks, backed by real playback. A thin adapter so
    ActionExecutor/CharacterOrchestrator never import sounddevice directly --
    they only ever see the Action vocabulary (play_sound/music_on/music_off)."""
    music = MusicLoop(LOUNGE_LOOP_PATH, volume=music_volume)

    def on_play_sound(name: str) -> None:
        path = os.path.join(SFX_DIR, name) if not os.path.isabs(name) else name
        play_once(path)

    return {
        "on_play_sound": on_play_sound,
        "on_music_on": music.start,
        "on_music_off": music.stop,
    }
