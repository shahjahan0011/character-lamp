# Character Lamp

A live, expressive character built around the supplied fictional 5-DOF
Luxo-Jr-style desk lamp, for the Live Character Robot challenge
([`CHALLENGE.md`](CHALLENGE.md), [`SUBMISSION.md`](SUBMISSION.md)).

It watches through your laptop camera, listens through your microphone,
speaks through your speaker, and moves/lights a simulated PyBullet body.
One continuous interaction covers all five challenge moments:
engagement, an expressive acknowledgment, spoken conversation, scene
memory (see an object, recall it later), and a goal-directed action
(find something in view, point at it, verify the result).

See [`docs/TECHNICAL_NOTE.md`](docs/TECHNICAL_NOTE.md) for architecture,
protocol, and measurements, and [`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md)
for a scripted walkthrough of one continuous demo covering all five moments.

## What's implemented vs. left out

Built and working, in the ~6-8 hour timebox: engagement via face
detection with hysteresis, an expressive acknowledgment (motion + light
+ chime), a persistent Gemini Live voice session (barge-in, tool use),
structured scene memory (remember/recall specific objects), a
goal-directed action workflow (observe → act → re-observe → verify,
enforced by a small state machine, not just prompted), a kinematic
PyBullet body respecting the URDF's joint limits, and an offline/mocked
demo + test suite that needs no camera, mic, or API key.

Intentionally left out: no 3D depth/pose estimation (pointing uses
approximate 2-D image coordinates, see the technical note); no on-device
speech or vision models (all cloud, via Gemini's free tier); no
multi-person handling (single simple face-detection gate); no packaging
beyond a plain Python entrypoint (see below).

**Missing supplied asset:** `robot/dummy-lamp.png` (referenced by
`CHALLENGE.md`) was never present in this repository's working tree or
git history at any point in this project — it appears the file was not
actually delivered alongside the URDF and STL. It is not fabricated as a
replacement; `.gitignore` carries an exception (`!robot/dummy-lamp.png`)
so it can be tracked immediately if it's ever supplied.

## Setup (Ubuntu 24.04 LTS target)

The target environment is Ubuntu 24.04 LTS, 4 CPU cores, 8 GB RAM, no
GPU, camera + mic/speaker via standard Linux audio, Wi-Fi. This is the
environment the setup steps below are written for. Development in this
repository was actually done on macOS — see "Development machine
(macOS)" below for what differs.

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip \
    libgl1 libglib2.0-0 portaudio19-dev ffmpeg

git clone <this-repo-url> character-lamp
cd character-lamp
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# edit .env, add GEMINI_API_KEY from https://aistudio.google.com/apikey
# (free tier, no credit card)
```

Ubuntu 24.04 ships Python 3.12 by default; PyBullet's prebuilt wheels
target cp310/cp311 (see `pyproject.toml`'s comment), so 3.11 is installed
explicitly above. If `python3.11` isn't available from the default repos
on a given Ubuntu point release, add the `deadsnakes` PPA first:

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
```

`libgl1`/`libglib2.0-0` are PyBullet's and OpenCV's runtime shared-library
dependencies on a minimal server/desktop install; `portaudio19-dev` is
needed to build `sounddevice`'s PortAudio binding.

### Run it

```bash
python scripts/run_character.py
```

Opens a PyBullet window, your camera, and your mic. Look at the camera
to engage; look away to disengage. See `docs/DEMO_SCRIPT.md` for what to
say once engaged. Ctrl+C to quit.

Useful flags:

```bash
python scripts/run_character.py --headless           # no GUI window
python scripts/run_character.py --offline            # deterministic mock demo: no camera/mic/network/API key needed
python scripts/run_character.py --list-audio-devices # list input/output devices, then exit
python scripts/run_character.py --force-engaged      # real camera, but skip face-detection gating (diagnostic)
python scripts/run_character.py --data-dir var        # where scene memory + metrics.jsonl are written (default: var)
```

Without `GEMINI_API_KEY` set, engagement/motion/light/music still work;
conversation, scene memory, and goal-directed action are disabled (the
launcher logs this explicitly rather than failing silently).

### Tests

```bash
pip install -e ".[dev]"
pytest              # 122 tests, fully mocked -- no camera/mic/network/API key touched
ruff check src/ tests/ scripts/
```

### Measurements

```bash
python scripts/run_character.py --offline --headless
python scripts/summarize_metrics.py var/metrics.jsonl
```

Every run appends latency/engagement/CPU/RSS events to
`<data-dir>/metrics.jsonl` (see `src/character/metrics.py`); the
summarizer prints the aggregate numbers cited in the technical note.

## Development machine (macOS)

This repo was built and tested on macOS, not Ubuntu. Two things differ
from the Ubuntu steps above:

- **PyBullet**: no prebuilt macOS wheel exists for the pinned version at
  the time of writing; it was built from source into the local venv.
  This is a one-time, machine-specific step and is *not* part of the
  Ubuntu target's install path above (Ubuntu uses the prebuilt
  manylinux wheel directly via `pip install`).
- **Camera/mic permission prompts**: macOS asks for Terminal/camera and
  Terminal/microphone permission the first time `run_character.py`
  opens them, via a system dialog rather than a Linux PipeWire/ALSA
  permission model.

Everything else (PyBullet body, Gemini Live session, scene memory, goal
workflow, tests) is identical code on both platforms.

## Troubleshooting

- **"Could not open camera index 0"**: another app may be holding the
  camera, or the index is wrong on a multi-camera machine — try
  `--camera-index 1`, etc.
- **No sound / wrong device**: run `--list-audio-devices` and confirm
  the system default output is the one you expect; `sounddevice` uses
  the OS default unless configured otherwise.
- **"No GEMINI_API_KEY found"**: copy `.env.example` to `.env` and add a
  key from https://aistudio.google.com/apikey — engagement/motion still
  work without one.
- **Vision quota exhausted mid-demo**: the Gemini vision free tier is
  rate-limited; `REOBSERVE_BACKOFF_S` in `src/character/fsm.py` backs
  off automatically after a rate-limit error rather than retry-looping.

## Privacy / cloud disclosure

Camera frames are sent to Google's Gemini vision API only when a scene
observation is actively requested (on engagement, on a ~45s interval
while engaged, or when a tool call asks for one) — never continuously
streamed. Microphone audio is streamed to Gemini's Live API only while
engaged, not mid-turn/mid-playback, and not at all while disengaged (see
`GeminiLiveClient.set_mic_gate` in `src/speech/live_client.py`). No
video/audio is persisted to disk; the only local persistence is
structured scene-memory text (`SceneMemory`, e.g. "red mug, ceramic, on
the desk") and the metrics log described above. All model inference
(vision, conversation, tool-use, TTS) happens in Google's cloud; nothing
runs a local ML model. See `docs/TECHNICAL_NOTE.md` for the full
protocol/trust-boundary discussion.
