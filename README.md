# Character Lamp

This is my take on the live-character challenge: a desk lamp that notices
when someone looks at it, turns and brightens to acknowledge them, holds a
spoken conversation, remembers objects it has seen, and can look or point
toward something after checking the scene again.

The important part for me was making those features feel like one character.
The same attention state controls the body, light, microphone, music, and
voice. When nobody is there, the lamp wanders gently with music playing. When
a face appears, it stops the music, chimes, flashes, turns, nods, and starts
listening. When attention leaves, it closes the mic before returning home.

The body is the supplied five-joint URDF running in PyBullet. Conversation is
a persistent Gemini Live session; scene descriptions use a separate Gemini
vision request. Object memory and all actuator commands remain local.

## Try it

The quickest confidence check needs no camera, microphone, API key, or display:

```bash
python scripts/run_character.py --offline --headless
```

That runs the real memory, tool, action, and goal-verification code against
three scripted camera observations. It should finish with `all 8 steps passed`.

To run the character for real:

```bash
cp .env.example .env
# Put your Gemini API key in .env
python scripts/run_character.py
```

Look toward the webcam and wait for the chime and nod before speaking. A good
first conversation is: “Hi—how are you?” Then place a distinct object in
view and try “Remember the red mug,” “What did you notice earlier?”, and
“Find the mug and point at it.” The complete walkthrough is in
[`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md); the recording-ready version is
[`docs/DEMO_VIDEO_SCRIPT.md`](docs/DEMO_VIDEO_SCRIPT.md).

If you only want to verify the Gemini side, without opening any local hardware:

```bash
python scripts/run_character.py --check-live
```

That checks more than the WebSocket handshake. It asks for a reply, completes
any blocking gesture tool call, and verifies that audio bytes and a completed
turn actually come back.

## Ubuntu 24.04 setup

I developed on macOS, but kept the runtime CPU-only and tested the project with
Python 3.11. On a clean Ubuntu laptop I would install it like this:

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip \
    libgl1 libglib2.0-0 portaudio19-dev ffmpeg

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env
```

If that Ubuntu image does not provide `python3.11`, use `uv python install
3.11` (or an equivalent Python 3.11 package source) and create the venv with
that interpreter. The application itself is launched from the repository root;
it is deliberately a single process rather than a service or container.

Useful launch options:

```bash
python scripts/run_character.py --headless
python scripts/run_character.py --force-engaged
python scripts/run_character.py --camera-index 1
python scripts/run_character.py --list-audio-devices
python scripts/run_character.py --log-level DEBUG
python scripts/run_character.py --mic-threshold 100
```

`--force-engaged` is the most useful camera diagnostic: it keeps the real
camera and microphone but removes face detection from the decision. `DEBUG`
logs the microphone gate, live RMS value, speech boundary, tool calls, and API
errors. If the debug log shows room noise permanently above the default RMS
threshold of 60, raise `--mic-threshold` until quiet sits below it and speech
sits above it. The same value can be saved as
`CHARACTER_LAMP_MIC_RMS_THRESHOLD` in `.env`.

## What is actually implemented

- Face-based engagement with separate engage/disengage hysteresis.
- A coordinated acknowledgement using motion, light, chime, and music.
- One persistent Gemini Live connection for low-latency native-audio dialogue.
- Local voice endpointing, mic/playback gating, interruption handling, and
  streamed audio mixing.
- Periodic and on-demand scene observations from the same webcam handle used
  for engagement.
- Structured object memory persisted to `var/scene_memory.json`.
- Seven schema-validated model tools. The model can ask for an observation,
  recall or store memory, move, change the light, gesture, and finish a goal;
  it never writes joint values directly.
- A locally enforced goal sequence: fresh observation → action → fresh
  post-action observation → verified finish.
- JSONL metrics for engagement, speech boundaries, tool calls, response
  latency, API errors, CPU, memory, and audio under/overruns.
- 124 hardware-free tests and a deterministic end-to-end offline demo.

The short architecture and tradeoff explanation is in
[`docs/TECHNICAL_NOTE.md`](docs/TECHNICAL_NOTE.md). I left the original
[`CHALLENGE.md`](CHALLENGE.md) and [`SUBMISSION.md`](SUBMISSION.md) untouched
because they are supplied evaluation material, not claims about my solution.

## Checks before a demo

```bash
python scripts/run_character.py --check-live
python scripts/run_character.py --offline --headless
pytest -q
ruff check .
```

For a hardware rehearsal, run with `--log-level DEBUG`, speak twice, ask for
one remembered object, and then inspect:

```bash
python scripts/summarize_metrics.py var/metrics.jsonl
```

A healthy voice run contains `speech_started`, `speech_committed`, and
`first_audio` events in that order. If engagement works but those are absent,
the problem is the microphone gate or local threshold—not Gemini. If
`--check-live` fails, the error is instead in the key, model availability, or
network path.

Scene vision and the small startup error-voice cache use the normal Gemini
`models.generate_content` endpoint. The earlier experimental Interactions
endpoint required different authentication for this key and was the concrete
reason object observation failed during review.

## Honest limits

Pointing is based on normalized 2-D image coordinates. There is no depth
camera, calibration between the laptop camera and the simulated lamp, or 3-D
pose estimate, so “pointing” means turning toward the object in image space.
The engagement detector follows one prominent frontal face and does not
identify people. The memory is structured text, not stored images, and there
is no long-term conversational transcript on disk.

The cloud dependency is also real. Microphone audio is sent to Gemini Live
only while the lamp is engaged and not playing its own audio. Individual
camera frames are sent when an observation is requested or during the 45-second
engaged refresh—not as a video stream. No audio or image is written to disk.
Free-tier quotas can still interrupt a long rehearsal, so vision backs off for
five minutes after a quota error.

The challenge references `robot/dummy-lamp.png`, but that file was not present
in this working tree or its history. I did not invent a replacement. The URDF,
STL, and all runtime behavior are present.
