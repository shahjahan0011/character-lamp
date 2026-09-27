# Demo script

One continuous interaction (~2-4 minutes) covering all five challenge
moments. Run `python scripts/run_character.py`, place one small object
(e.g. a red mug) in view of the camera, and follow along. Actual model
replies vary — this is a guide, not a transcript.

For a timed, recording-ready shot list with exact dialogue and a post-take
checklist, use [`DEMO_VIDEO_SCRIPT.md`](DEMO_VIDEO_SCRIPT.md).

## Setup

- A well-lit room, laptop camera pointed at where you'll sit.
- One distinct small object (a mug, a toy, a bottle) within the
  camera's view but off to one side.
- `GEMINI_API_KEY` set in `.env` (conversation/memory/goal-action need
  it; engagement/motion/light/music work without it).

## 1. Engagement (moment 1)

Sit down facing the camera. Within about a second of consecutive frames
detecting your face, the lamp: plays an engage chime, flashes its light
twice, brightens, and turns/nods toward you (`_on_engage` in
`src/character/fsm.py`). This is the acknowledgment moment (moment 2) —
motion + light + sound in one intentional reaction, not a generic "on"
state.

Look away for a couple of seconds — the lamp dims back down and returns
to its idle wander (occasional small motions, background music). Look
back to re-engage before continuing.

## 2. Spoken interaction (moment 3)

Say something simple: *"Hi there, how are you?"* A local silence
detector commits your utterance about 400ms after you stop talking; the
light shifts to a cool "thinking" color with a soft hum while the reply
is in flight, then the lamp answers out loud (Gemini Live, persistent
session — no per-turn reconnect) and reacts with a matching gesture
(`nod`/`curious`/`excited`/etc., chosen by the model per turn).

Try interrupting it mid-reply — barge-in is supported (`interrupted`
event stops playback immediately).

## 3. Scene memory (moment 4)

Ask: *"Can you remember that mug for me?"* This triggers
`request_observation(purpose="object_memory")` → a fresh camera frame →
Gemini vision returns a grounded description (label, color, approximate
position) → `remember_object` stores it. The lamp should confirm
verbally that it noted the object.

Later in the conversation (doesn't need to be immediately after), ask:
*"What did you see earlier?"* or *"Do you remember the mug?"* — this
calls `recall_memory`, which searches the stored record by text overlap
and answers only from what's actually stored, not by guessing.

## 4. Goal-directed action (moment 5)

Ask: *"Can you find the mug and point at it?"* The lamp: takes a fresh
`request_observation(purpose="goal_planning")` (never acts on a stale
observation, even if it already remembers the object), converts the
returned 2-D image coordinate to pan/tilt and calls
`look_at_image_point` to physically point/look at it, then takes a
second `request_observation(purpose="goal_verification")` to check the
result before calling `finish_goal`. `GoalCoordinator` enforces this
exact ordering locally — a model that tries to skip re-observing or
claim success against a stale/wrong observation is rejected, not
trusted blindly.

This moment naturally combines with the acknowledgment moment (light +
motion) and can be repeated with a different phrasing (*"look at the
mug"*, *"point to what's on the left"*) to show it isn't a single
hardcoded path.

## Wrap-up

Look away from the camera to disengage — light dims, mic and Live audio
gate close immediately (before any further motion), and the lamp
returns to idle wandering with background music. This closes the loop
back to moment 1's engagement/disengagement pair.

## If something goes wrong live

The lamp speaks a short spoken error (not silence) if the Live
connection drops or a turn times out (`src/speech/error_speech.py`,
demonstrating the "speak basic errors" behavior) rather than just
going quiet.
