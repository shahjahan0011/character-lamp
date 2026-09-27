# Technical Note — Character Lamp

*Target length: 2 pages. See README.md for setup and DEMO_SCRIPT.md for the walkthrough.*

## Architecture and data flow

```
 Camera ──► EngagementWatcher ──► CharacterOrchestrator ──► ActionExecutor ──► PyBullet (kinematic body)
 (thread)   (Haar-cascade face,        │      ▲                   │
             hysteresis, 10Hz)         │      │                   └──► AudioMixer ──► speaker
                                       │      │                        (speech+SFX+music, ducked)
 Mic ────► LiveMicStreamer ───────────►│      │
 (thread)   (local VAD, silence-       │      │
             commit)                  ▼      │
                              GeminiLiveClient (own asyncio thread)
                              persistent bidirectional session
                                       │      ▲
                    tool_call/audio_chunk      tool_result/mic audio/text nudge
                                       │      │
                                       ▼      │
                                  ToolGateway ──► ObservationRegistry / GoalCoordinator / SceneMemory
                                       │
                                       ▼
                              SceneObserver (own thread) ──► Gemini vision API (schema-constrained JSON)
```

Four background threads (camera, mic, Live session, scene observer) each
expose a thread-safe polled interface; **only the main thread ever
touches PyBullet or dispatches an `Action`** — `CharacterOrchestrator.tick()`,
called at 10Hz, is the single point where all four streams are read and
turned into body/light/audio commands. This mirrors the constraint a
real embedded lamp would have (one control loop owning the actuators)
and avoids cross-thread PyBullet calls, which are not thread-safe.

## Protocol and trust boundary

Two closed vocabularies separate "what the model may decide" from "what
the body can do":

- **`Action`** (`src/protocol/models.py`) — `look_at`, `point_at`, `nod`,
  `set_light`, `play_sound`, `home`, etc. The model never names a joint,
  torque, or file path; `ActionExecutor` is the only thing that knows
  those.
- **Gemini Live tools** (`src/protocol/tools.py`) — 7 Pydantic models,
  each `extra="forbid"` with bounded fields (e.g. `x`/`y` ∈ [0,1]),
  generated into Gemini's function-calling schema. All run with
  `behavior: BLOCKING`, so the model waits for a real result (confirmed
  live) instead of guessing ahead — essential for enforcing goal
  ordering (see below). `ToolGateway` validates and dispatches every
  call; a malformed or out-of-order call returns a structured `{"ok":
  false, "error": ...}` rather than crashing the turn.

Cloud usage: Gemini's vision, Live (speech+tool-use), and TTS APIs
(free tier, Flash models) do all inference — no local model. A camera
frame is sent only when a specific observation is requested (on
engage, ~45s while engaged, or a tool call), never streamed
continuously; mic audio streams only while engaged and no
turn/playback is in flight. See README.md's privacy section.

## Model-to-action flow (goal-directed action)

`GoalCoordinator` enforces, independent of what the model claims: a
fresh `goal_planning` observation must exist before any action; at
least one action must be recorded; a `goal_verification` observation
must be captured *after* that action (by monotonic timestamp); and
`finish_goal` must reference exactly that verification observation.
One inconclusive retry is allowed before the goal is marked `failed`.
This turns "the LLM says it worked" into a checkable invariant — a
model that skips re-observing, or claims success against a stale
observation, is rejected locally, not trusted.

`look_at_image_point` converts a normalized 2-D image coordinate to
pan/tilt (`image_point_to_pan_tilt`, `src/body/executor.py`) — this is
approximate image-space pointing, not 3-D localization (no depth
sensing exists in this system).

## Simulation and physical reasoning

PyBullet loads the supplied URDF unmodified and drives joints
*kinematically* (`resetJointState`, not motor/dynamics control) — the
URDF defines no `<transmission>`, so there's nothing for a PD controller
to actuate. `TrajectoryPlayer` still respects the URDF's own per-joint
velocity limits: moves are smoothstep-interpolated (zero velocity at
both ends, avoiding jerk at move boundaries), with duration scaled by
1.5× (`SMOOTHSTEP_PEAK_FACTOR`) because a smoothstep's *peak*
instantaneous velocity is 1.5× its average — computing duration from
plain `distance / max_velocity` would silently violate the limit at
the midpoint of every move. Multi-joint moves share one synchronized
duration so a "return home" gesture settles as one motion, not several
joints arriving at different times.

## Deployment

Plain Python process (`python scripts/run_character.py`), no daemon/
container — matches a single always-on lamp, not a multi-tenant
service. `pyproject.toml` pins `requires-python>=3.11,<3.13` because
PyBullet ships prebuilt wheels for cp311 but not cp312+, and Ubuntu
24.04's default `python3` is 3.12 (see README.md's `deadsnakes` PPA
step). No GPU/CUDA dependency, matching the target's constraints.

## Measurements

**Response latency (first audio byte after utterance end), real API,
warm connection** — measured manually during development against the
live Gemini API (not fabricated; see `src/speech/live_client.py`'s
module docstring for the original measurement notes): `gemini-3.8-live`
landed **~1.2s**; the documented fallback model,
`gemini-2.5-flash-native-audio-latest`, landed **~1.7–2.2s** across
repeated real turns. A fresh/cold connection's *first* turn occasionally
took 10+ seconds (one outlier in ~10 real calls) — this is why the
client connects once at startup and stays connected for the app's whole
lifetime, rather than reconnecting per turn. These are small-sample,
manual dev-time measurements, not a large statistical benchmark.

**CPU / memory** — measured with this repo's own instrumentation
(`src/character/metrics.py`, real `psutil` samples, JSONL, see
`scripts/summarize_metrics.py`), on the macOS development machine (not
the Ubuntu target — see "Known limitations"), in `--offline --headless`
mode (no camera/mic/network — simulation + audio mixer + goal/memory
workflow only): steady-state idle 10Hz polling loop over 20s measured
**peak RSS 138 MB**, **mean process CPU ~0.2% of one core**. Startup
(PyBullet + numpy + OpenCV import and init) is the dominant cost, not
the steady-state loop. These numbers exclude the camera/mic capture
threads and any live network call, so they are a floor, not a full
live-session measurement.

**Engagement reliability** — **not measured**: this requires labeled
trials with a real person in front of a real camera, which this
development/verification environment could not perform (no interactive
camera access in the agent sandbox used to build this). The intended
procedure: face the camera, log `EngagementWatcher`'s `changed_at`
transitions against ground truth for ~20 trials, and report true/false
engagement rate and median engage/disengage latency. This is left as an
explicit gap rather than a fabricated number — see Known limitations.

## Known limitations

- Engagement reliability is unmeasured (see above) — the mechanism
  (Haar-cascade face detection with a consecutive-frame hysteresis) is
  implemented and used live in earlier development, but no formal trial
  set was run.
- CPU/RSS measurements are from macOS, not the Ubuntu 24.04 target, and
  exclude camera/mic capture and any live network call.
- `robot/dummy-lamp.png` (referenced by `CHALLENGE.md`) was never
  present in this repository's history — reported honestly rather than
  fabricated (see README.md).
- Pointing is 2-D image-space only; no depth/3-D localization.
- Single simple engagement gate — no multi-person handling, no
  identity/re-engagement memory across sessions.
- Vision/TTS run on Gemini's free tier, which is rate-limited; a
  sustained demo can exhaust a daily quota (`REOBSERVE_BACKOFF_S`
  backs off automatically rather than retry-looping).
