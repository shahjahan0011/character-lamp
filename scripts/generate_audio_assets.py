"""Synthesizes the project's placeholder audio assets -- an engagement
chime and an ambient idle music loop -- entirely in Python. No downloaded
or recorded material, so there's no licensing question at all: we own
every sample. Documented in NOTES.md as a known limitation (a produced
track would sound better; this is what fit the time budget).

Usage: .venv/bin/python scripts/generate_audio_assets.py
"""

import os

import numpy as np
import soundfile as sf

SR = 44100
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")


def _fade(signal: np.ndarray, fade_samples: int) -> np.ndarray:
    fade_samples = min(fade_samples, len(signal) // 2)
    ramp = np.linspace(0.0, 1.0, fade_samples)
    signal = signal.copy()
    signal[:fade_samples] *= ramp
    signal[-fade_samples:] *= ramp[::-1]
    return signal


def make_chime() -> np.ndarray:
    """A short two-note ascending bell -- "notice" cue for engagement."""
    notes_hz = [880.0, 1318.51]  # A5 -> E6, a clean perfect-fifth "ding-dong"
    note_dur = 0.35
    gap = 0.08
    out = np.zeros(int(SR * (note_dur * 2 + gap)))
    for i, freq in enumerate(notes_hz):
        t = np.linspace(0, note_dur, int(SR * note_dur), endpoint=False)
        # A couple of harmonics plus an exponential decay envelope reads as
        # a soft bell rather than a flat sine beep.
        tone = 0.6 * np.sin(2 * np.pi * freq * t)
        tone += 0.25 * np.sin(2 * np.pi * freq * 2 * t)
        tone += 0.1 * np.sin(2 * np.pi * freq * 3 * t)
        envelope = np.exp(-t * 6.0)
        tone *= envelope
        start = int(SR * i * (note_dur + gap))
        out[start : start + len(tone)] += tone
    out = out / max(np.abs(out).max(), 1e-9) * 0.8
    return _fade(out, int(SR * 0.02))


def make_lounge_loop(duration_s: float = 16.0) -> np.ndarray:
    """A slow, soft ambient pad loop -- ii-V-I-vi style chord bed with a
    gentle filtered-noise "shaker" pulse, meant to sit quietly in the
    background while nobody's engaged. Loops seamlessly (matching phase at
    the loop boundary -- each chord is an integer number of beats)."""
    bpm = 70
    beat = 60.0 / bpm
    chord_beats = 4
    chord_dur = beat * chord_beats
    # ii - V - I - vi (A minor / C major feel), voiced as simple triads.
    chords_hz = [
        [220.00, 261.63, 329.63],  # A3 C4 E4  (Am)
        [196.00, 246.94, 293.66],  # G3 B3 D4  (G)
        [261.63, 329.63, 392.00],  # C4 E4 G4  (C)
        [220.00, 261.63, 349.23],  # A3 C4 F4  (Fmaj7-ish / Am add6)
    ]
    n_samples = int(SR * chord_dur * len(chords_hz))
    out = np.zeros(n_samples)
    t_chord = np.linspace(0, chord_dur, int(SR * chord_dur), endpoint=False)
    # Slow attack/release per chord so changes are a soft swell, not a click.
    envelope = np.sin(np.pi * t_chord / chord_dur) ** 0.7

    for i, chord in enumerate(chords_hz):
        pad = np.zeros_like(t_chord)
        for freq in chord:
            pad += np.sin(2 * np.pi * freq * t_chord)
            pad += 0.3 * np.sin(2 * np.pi * freq * 2 * t_chord)  # gentle octave shimmer
        pad = pad / len(chord)
        pad *= envelope
        start = int(SR * chord_dur * i)
        out[start : start + len(pad)] += pad

    # A quiet, slow filtered-noise pulse on each beat -- reads as a soft
    # brushed-shaker texture under the pads, common in lounge/lofi beds.
    rng = np.random.default_rng(seed=7)
    noise = rng.standard_normal(n_samples)
    # Simple one-pole low-pass to take the harsh edge off the noise.
    filtered = np.zeros_like(noise)
    alpha = 0.05
    prev = 0.0
    for i in range(len(noise)):
        prev = alpha * noise[i] + (1 - alpha) * prev
        filtered[i] = prev
    beat_samples = int(SR * beat)
    pulse_env = np.zeros(n_samples)
    for start in range(0, n_samples, beat_samples):
        length = min(int(beat_samples * 0.15), n_samples - start)
        decay = np.exp(-np.linspace(0, 8, length))
        pulse_env[start : start + length] = decay
    out += filtered * pulse_env * 0.05

    out = out / max(np.abs(out).max(), 1e-9) * 0.5
    return _fade(out, int(SR * 0.3))


def main() -> None:
    sfx_dir = os.path.join(OUT_DIR, "sfx")
    music_dir = os.path.join(OUT_DIR, "music")
    os.makedirs(sfx_dir, exist_ok=True)
    os.makedirs(music_dir, exist_ok=True)

    chime_path = os.path.join(sfx_dir, "engage_chime.wav")
    sf.write(chime_path, make_chime(), SR)
    print(f"Wrote {chime_path}")

    loop_path = os.path.join(music_dir, "lounge_loop.wav")
    sf.write(loop_path, make_lounge_loop(), SR)
    print(f"Wrote {loop_path}")


if __name__ == "__main__":
    main()
