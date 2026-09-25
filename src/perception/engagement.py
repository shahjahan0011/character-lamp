"""Webcam-based engagement detection: is a person currently looking toward
the lamp?

Runs in its own background thread (see EngagementWatcher) so the body
executor's blocking gesture calls never stall frame capture. Uses a Haar
cascade -- it ships with opencv-python (no model download) and, since it
only reliably fires on roughly-frontal faces, doubles as a cheap "looking at
the camera" proxy rather than just "a face exists somewhere in frame".

This module only produces a signal (EngagementState) and a rough horizontal
position of the face in frame -- it has no idea what a joint or an Action
is. The FSM/orchestrator (later) decides what to do with that signal.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2


@dataclass
class EngagementState:
    engaged: bool
    face_x_frac: float = 0.0  # -1 (frame left) .. +1 (frame right), 0 = center
    changed_at: float = 0.0  # time.time() of the last engaged/disengaged flip


class EngagementWatcher:
    """Owns the camera and a background thread. Call start()/stop() around
    its lifetime and read the latest state with get_state() from any thread."""

    def __init__(
        self,
        camera_index: int = 0,
        frames_to_engage: int = 3,
        frames_to_disengage: int = 15,
        detect_hz: float = 8.0,
        detect_width: int = 320,
    ):
        # Hysteresis: a few steady "face seen" frames to engage (fast, since
        # noticing someone should feel immediate) but many more steady
        # "no face" frames to disengage (slow, so a blink or a half-second
        # head turn doesn't make the lamp flicker in and out of attention).
        # Both counts are in units of *detections*, not camera frames -- see
        # detect_hz.
        self._frames_to_engage = frames_to_engage
        self._frames_to_disengage = frames_to_disengage
        # Haar detection on a full-resolution frame is expensive enough to
        # peg a CPU core if run flat-out at the camera's native frame rate,
        # which visibly starves everything else (PyBullet's renderer
        # included) of CPU. Two independent cheap fixes: cap the detection
        # rate, and detect on a small downscaled copy of the frame (the
        # fractional face position we need doesn't care about resolution).
        self._detect_interval_s = 1.0 / detect_hz
        self._detect_width = detect_width

        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self._cascade.empty():
            raise RuntimeError("Failed to load Haar cascade -- opencv-python install is incomplete")

        self._camera_index = camera_index
        self._capture: cv2.VideoCapture | None = None

        self._lock = threading.Lock()
        self._state = EngagementState(engaged=False, changed_at=time.time())
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        self._capture = cv2.VideoCapture(self._camera_index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Could not open camera index {self._camera_index}")
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._capture is not None:
            self._capture.release()

    def get_state(self) -> EngagementState:
        with self._lock:
            return EngagementState(**vars(self._state))

    def _run(self) -> None:
        consecutive_face = 0
        consecutive_absent = 0
        next_detect_at = 0.0

        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.05)
                continue

            now = time.time()
            if now < next_detect_at:
                # Still capture (so the driver's buffer doesn't back up),
                # but skip the expensive detection pass and don't spin.
                time.sleep(0.01)
                continue
            next_detect_at = now + self._detect_interval_s

            frame_h, frame_w = frame.shape[:2]
            scale = self._detect_width / frame_w
            small = cv2.resize(frame, (self._detect_width, int(frame_h * scale)))

            face_x_frac = 0.0
            found = False
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
            if len(faces) > 0:
                # Largest face = closest/most prominent person in frame.
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                center_x = x + w / 2
                face_x_frac = (center_x / small.shape[1]) * 2 - 1  # -1 .. +1
                found = True

            if found:
                consecutive_face += 1
                consecutive_absent = 0
            else:
                consecutive_absent += 1
                consecutive_face = 0

            with self._lock:
                was_engaged = self._state.engaged
                now_engaged = was_engaged
                if not was_engaged and consecutive_face >= self._frames_to_engage:
                    now_engaged = True
                elif was_engaged and consecutive_absent >= self._frames_to_disengage:
                    now_engaged = False

                self._state = EngagementState(
                    engaged=now_engaged,
                    face_x_frac=face_x_frac if found else self._state.face_x_frac,
                    changed_at=time.time() if now_engaged != was_engaged else self._state.changed_at,
                )
