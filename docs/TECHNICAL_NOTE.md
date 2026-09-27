# Technical note — Character Lamp

I built the lamp around one simple rule: attention should be shared by every
part of the character. Face detection does not merely trigger an animation;
it decides when the music stops, when the microphone may stream, when replies
may play, and when the lamp should return home. That is what keeps the demo
from feeling like separate vision, chatbot, and robot samples.

## The system I ended up with

```text
 webcam ─► EngagementWatcher ───────┐
           face + hysteresis        │
                                    ▼
 microphone ─► LiveMicStreamer ─► CharacterOrchestrator (10 Hz) ─► ActionExecutor ─► PyBullet
                  │                 │          ▲                         │
                  ▼                 │          │                         └─► light / audio mixer
          Gemini Live session ──────┤          │
          voice + tool calls        ▼          │ tool results
                               ToolGateway ─────┘
                                    │
 webcam frame ─► SceneObserver ─────┼─► ObservationRegistry
                  Gemini vision     ├─► SceneMemory (local JSON)
                                    └─► GoalCoordinator
```

Camera capture, microphone capture, the Live socket, and scene observation run
on background threads. The main 10 Hz loop is the only owner of PyBullet and
the only place that dispatches body actions. I chose this plain threaded shape
because camera/audio libraries are blocking, Gemini Live is async, and
PyBullet calls are safest when they stay on one thread. It is less fashionable
than an event bus, but much easier to trace during a live demo.

Conversation originally used three serial steps: record a whole utterance,
ask a model for text, then make a second TTS call. It was slow and eventually
went silent against a small TTS quota. The current version keeps one Gemini
Live native-audio session open for the process lifetime. Audio chunks go from
the network thread straight into the mixer so a blocking body gesture cannot
starve playback. The main loop still receives lightweight “audio started” and
“turn complete” events for state and lighting.

## Where I trust the model—and where I do not

The model sees seven tools, each generated from a strict Pydantic argument
model. Coordinates and brightness are bounded, extra fields are rejected, and
tool calls return structured success or error objects. Stateful tools block
until their real local result is available. A purely expressive gesture is
non-blocking so speech can begin while the body reacts instead of waiting for
the animation to finish.

The model never sees joint names, file paths, motor effort, or arbitrary code.
It chooses from semantic actions such as `perform_gesture`, `set_light`, and
`look_at_image_point`. `ToolGateway` validates those choices, and
`ActionExecutor` is the only layer that translates them into the supplied
five-joint body.

I was especially careful with the challenge's goal-directed action. Prompting
the model to “look again” was not enough. `GoalCoordinator` rejects completion
unless it can prove this order locally:

```text
fresh planning observation → at least one action → newer verification observation → finish_goal
```

Observation IDs and monotonic timestamps enforce that ordering. An old memory
can help conversation, but it cannot authorize a new movement. One retry is
allowed after an inconclusive check; otherwise the goal fails rather than the
lamp claiming success.

`look_at_image_point` maps a normalized camera coordinate to bounded pan and
tilt. This is intentionally described as approximate image-space pointing.
Without depth or camera-to-robot calibration, calling it 3-D localization
would be misleading.

## Body and physical choices

The supplied URDF has joint geometry and limits but no transmissions or motor
controller. I therefore drive it kinematically with PyBullet
`resetJointState`. Every move still respects the URDF velocity limits.
Smoothstep interpolation removes abrupt starts and stops; its peak speed is
1.5 times its average, so move duration is multiplied by 1.5 rather than using
the tempting but incorrect `distance / limit` calculation. Multi-joint poses
share the slowest required duration so they arrive as one gesture.

The three fixed semantic links (light, camera, speaker) now have small,
plausible inertial values. They do not change controlled-joint behavior, but
they keep the URDF physically complete and prevent PyBullet from inventing a
unit mass for each link. Idle motion is a small sub-second glance; the original
full-body wander blocked the attention loop for 3–6 seconds.

The acknowledgement is deliberately local and deterministic: stop music,
chime, flash, turn toward the face, nod, and settle to a warm light. A network
round trip at that moment made the lamp feel inattentive. Language and gesture
selection remain model-driven once conversation begins.

## Deployment and data

This is one Python process for an Ubuntu 24.04 laptop with four CPU cores, 8 GB
RAM, camera, standard audio, and Wi-Fi. It has no CUDA or local-model
dependency. Python 3.11 is the tested interpreter. The README contains the
exact setup and four checks I would run before presenting.

Gemini receives 16 kHz microphone PCM while the lamp is engaged. It receives a
single camera frame on engagement, on an explicit tool request, and at a
45-second engaged refresh. Frames and audio are not persisted. The only saved
content is a small JSON list of object descriptions plus JSONL operational
metrics. This is a conscious cloud tradeoff: much less local compute and a
better voice interaction, in exchange for Wi-Fi, API availability, and quota
dependence.

## Evidence, including the awkward parts

- The automated suite contains **124 tests**. It covers the state machine,
  actions, schemas, observation freshness, memory, goal ordering, and the
  async tool-result path without touching hardware or the network.
- The offline end-to-end run completes all eight scripted stages using the
  production gateway, goal coordinator, memory, executor, and simulator.
- A real Gemini Live probe on 26 September 2026 connected to
  `gemini-3.8-live`, completed a `perform_gesture` call, returned audio, and
  closed the turn. The latest hardware log measured warm first-audio latency
  at **2.22–2.50 seconds**, with one **7.21-second** cold/tool outlier. These
  are small development samples, not a benchmark. After removing the hidden
  ambient text turn and making expressive gestures asynchronous, the final
  API self-check reached first audio in **1.19 seconds**.
- A real structured-vision probe through the final API path returned the
  synthetic red object at normalized `(0.71, 0.45)`. A separate error-voice
  probe returned a valid **67,506-byte RIFF/WAV**. Both previously used an
  experimental endpoint that rejected this API key with HTTP 401; they now use
  the supported `models.generate_content` path.
- The saved attempted hardware run recorded three engagements and one
  disengagement but no completed voice turn. That was useful: it localized the
  observed “no reply” problem before the model boundary. I added explicit
  metrics for mic start/commit/tool/error events, a debug RMS meter, an
  adjustable mic threshold, and a hardware-free `--check-live` command instead
  of hiding the gap behind a claim that everything had been tested.
- That same mixed macOS log (727 seconds, including GUI/camera runs) reports
  **122% mean process CPU** after discarding the first sample and **633 MB peak
  RSS**. One core equals 100%, so it uses roughly 1.2 of the four target cores.
  This is more representative than the earlier headless-only floor, but it is
  still not an Ubuntu measurement.

Engagement accuracy is not formally measured. The mechanism has asymmetric
hysteresis—three positive detections to engage, fifteen misses to disengage—
but I do not have labeled camera trials from the target laptop. Before a final
demo I would run 20 approach/look-away trials and report successful engages,
false drops, and median transition time from the existing event log.

## What I would improve next

First, I would calibrate microphone RMS and camera-to-lamp pointing on the
actual presentation laptop. Next I would replace Haar detection with a more
robust lightweight face-or-attention signal, while preserving the same local
engagement contract. With more time and hardware, depth or a calibrated camera
transform would turn the current “look toward it” action into defensible 3-D
pointing. Multi-person attention, offline inference, and a richer memory are
valuable, but I would not add them before the core five-minute interaction is
rehearsed and measured end to end.
