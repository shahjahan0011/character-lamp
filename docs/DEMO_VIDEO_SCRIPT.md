# Submission video script

The safest submission is one short introduction followed by one genuinely
continuous interaction. Aim for **3–4 minutes**. Do not try to demonstrate
every debug feature; make the five required moments unmistakable.

## Before recording

Use a red mug or another large, single-color object. Put it on the right side
of the webcam frame, with your face near the center and enough light for both
to be clear. Use headphones only for rehearsal—the final take should let the
camera hear the real laptop speaker, but keep speaker volume moderate so the
mic does not echo.

Run this checklist before opening the recorder:

```bash
python scripts/run_character.py --check-live
python scripts/run_character.py --offline --headless
pytest -q
python scripts/run_character.py --log-level DEBUG --data-dir var/rehearsal
```

In the rehearsal log, confirm `live mic gate: open`, then speak once and look
for `speech started`, `speech ended`, and `first audio`. If quiet-room RMS is
above 60, choose a threshold between the quiet and speaking values. Use that
same `--mic-threshold` in the final command.

For the final take, use a fresh data directory so the recall cannot be coming
from an old run:

```bash
python scripts/run_character.py --data-dir var/demo-take-01
```

Do not use `--force-engaged` in the submission—it bypasses the first required
behavior. Use `--log-level INFO`; DEBUG is useful for rehearsal but visually
noisy.

### Recording layout

- Make the PyBullet window the largest element.
- Include a webcam inset showing you and the red mug. OBS can capture both.
- Keep the terminal visible only as a narrow strip if you want proof of the
  live connection. Do not show `.env` or any API key.
- Enable Do Not Disturb and close unrelated windows.
- Start recording before launching the app. Trim only dead time before launch
  and after shutdown; keep the central interaction uncut.

## Spoken script and timing

### 0:00–0:20 — What this is

Say to the camera:

> “This is Character Lamp, built around the supplied five-degree-of-freedom
> URDF. It uses my laptop camera, microphone, and speaker. I’ll show one
> continuous interaction: engagement, character response, conversation,
> scene memory, and a vision-grounded action with verification.”

Launch the app. Stay outside the camera view for a few seconds so the lamp is
visibly idle and the quiet background music is audible.

### 0:20–0:45 — Engagement and acknowledgement

Move into view and look toward the webcam. Do not speak yet. Let the viewer see
the music stop, chime play, light flash, and lamp turn and nod.

After the reaction, say:

> “That response was local and immediate—the face signal controls attention,
> music, light, motion, and whether the microphone is allowed to stream.”

This clearly covers challenge moments 1 and 2.

### 0:45–1:15 — Spoken conversation

Face the webcam and say, at a normal pace:

> “Hi, Lamp. How are you feeling today?”

Stop completely and wait. Do not fill the silence; the local endpoint detector
needs a clean pause. Let the entire answer and gesture finish.

Optional second line, only if the first exchange was clean:

> “What makes you different from an ordinary desk lamp?”

This is challenge moment 3. One good exchange is stronger than several rushed
ones.

### 1:15–1:55 — Observe and remember

Point briefly toward the mug without covering it and say:

> “Please look at the scene and remember the red mug on my right.”

Wait for the confirmation. The lamp is taking a fresh frame, receiving a
structured observation, and storing the selected object locally.

Move the mug out of the webcam frame. Then ask:

> “What object did you notice earlier, and what color was it?”

Wait for the full answer. Removing the mug makes it visually clear that the
answer comes from memory, not the current frame. This is challenge moment 4.

### 1:55–2:50 — Goal-directed action

Put the mug back in view, this time clearly off-center. Say:

> “Find the red mug and point toward it. Check your work before you tell me
> you’re done.”

Keep both yourself and the mug still. This sequence can take longer than an
ordinary reply because it intentionally performs:

1. a fresh planning observation;
2. a bounded image-space pointing action;
3. a second, newer observation;
4. local verification through `finish_goal`.

Let the motion and final spoken result finish. Do not narrate over it. This is
challenge moment 5 and the strongest technical moment in the video.

### 2:50–3:15 — Disengagement and close

Look away and step out of view. Hold the shot long enough to show the light
dim, body return home, microphone close, and background music resume.

End with:

> “The language model chooses from semantic tools, but local code validates
> every argument, owns the joints, and refuses goal completion without a newer
> post-action observation. Object descriptions and metrics are stored locally;
> images and audio are not.”

Stop the app with Ctrl+C, then stop recording.

## What to check after the take

Run:

```bash
python scripts/summarize_metrics.py var/demo-take-01/metrics.jsonl
```

Keep the take only if all of these are true:

- engagement and disengagement are both visible;
- the acknowledgement uses motion, light, sound, and the music transition;
- at least one spoken reply is understandable;
- the removed mug is recalled correctly;
- the mug is put back before the goal request;
- the lamp visibly points and gives a final result after re-observing;
- the metrics contain `speech_started`, `speech_committed`, `first_audio`, and
  the relevant tool calls;
- no API key, `.env` content, unrelated notification, or private window is
  visible.

If a step fails, stop and make a new take with a new directory such as
`var/demo-take-02`. Do not splice together separate interactions and present
them as one continuous run.

## Fast failure guide during rehearsal

- **No engagement:** improve frontal lighting or adjust camera index. Use
  `--force-engaged` only to isolate the rest of the system.
- **Engaged, but speech is never detected:** inspect DEBUG RMS and adjust
  `--mic-threshold`.
- **Speech commits, but no reply:** run `--check-live`; check the terminal for
  a Live error.
- **Object request fails:** keep the object large, well lit, and separated from
  clutter; check vision quota/API errors.
- **Pointing goes the wrong direction:** swap the object to the opposite side
  only during rehearsal and verify the webcam is not horizontally mirrored.
- **Audio crackles:** check the metrics for `audio_xrun`; close heavy apps and
  record at 30 fps rather than 60 fps.
