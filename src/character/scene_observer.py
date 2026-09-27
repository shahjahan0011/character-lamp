"""Runs the scene-description Gemini call (demo moment 4) on its own
background thread, so a several-second vision call never blocks
CharacterOrchestrator.tick().

Used to also process spoken-utterance replies (this class was named
DialogueWorker) before conversation moved to a persistent Gemini Live
session (see live_client.py) -- that round trip doesn't exist as a
separate background job anymore, so this class now only ever does one
kind of job: describe a frame, remember what it saw.
"""

from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from src.character.memory import SceneMemory
from src.speech import vision
from src.speech.gemini_client import GeminiStreamError, call_with_retry
from src.speech.vision import ObservedObject

_RATE_LIMIT_CODES = frozenset({"rate_limit_exceeded", "quota_exceeded", "too_many_requests"})


@dataclass
class _ObserveJob:
    frame: np.ndarray


class SceneObserver:
    def __init__(
        self,
        memory: Optional[SceneMemory] = None,
        on_debug: Optional[Callable[[str], None]] = None,
    ):
        self._memory = memory
        self._on_debug = on_debug or (lambda msg: None)
        self._input: "queue.Queue[_ObserveJob]" = queue.Queue()
        self._busy = threading.Event()

        self._result_lock = threading.Lock()
        self._new_observation: Optional[tuple[list[ObservedObject], bool]] = None
        self._failed = False
        self._rate_limited = False

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit_observation(self, frame: np.ndarray) -> bool:
        """Queues a scene-description job for the given frame -- called
        once per engagement and then periodically while engaged (see
        CharacterOrchestrator._check_reobserve). Refuses (and drops the
        frame) if a previous observation is still in flight; the next
        periodic trigger will just try again with a fresher frame."""
        if self._busy.is_set():
            self._on_debug("scene observer: busy, skipping this observation")
            return False
        self._input.put(_ObserveJob(frame=frame))
        return True

    def is_busy(self) -> bool:
        return self._busy.is_set()

    def pop_new_observation(self) -> Optional[tuple[list[ObservedObject], bool]]:
        """Non-blocking poll from the main thread; returns and clears
        (objects, changed) for whatever the most recently finished
        observation saw, if any -- including an empty objects list for
        "looked, saw nothing notable", so a caller can tell that apart
        from "no observation has finished yet" (None). `changed` is
        SceneMemory.remember()'s return value: whether this observation
        actually added or updated anything, vs. re-confirming an
        unchanged scene."""
        with self._result_lock:
            result, self._new_observation = self._new_observation, None
        return result

    def pop_failure(self) -> bool:
        with self._result_lock:
            failed, self._failed = self._failed, False
        return failed

    def pop_rate_limited(self) -> bool:
        """Non-blocking poll: whether the last failure was specifically a
        rate/quota limit (vs. some other error) -- lets the caller back off
        the next re-observation much further out instead of retrying on
        the normal cadence and hitting the same wall again immediately.
        Confirmed live: re-observing every ~15s exhausted a 20-requests/day
        vision-model quota within minutes; a caller needs to know *why* it
        failed to react proportionately."""
        with self._result_lock:
            rate_limited, self._rate_limited = self._rate_limited, False
        return rate_limited

    def _run(self) -> None:
        while True:
            job = self._input.get()
            self._busy.set()
            try:
                self._process(job)
            except Exception as exc:  # noqa: BLE001 -- one bad observation shouldn't kill the worker
                self._on_debug(f"scene observer error:\n{traceback.format_exc()}")
                is_rate_limit = isinstance(exc, GeminiStreamError) and exc.code in _RATE_LIMIT_CODES
                with self._result_lock:
                    self._failed = True
                    self._rate_limited = is_rate_limit
            finally:
                self._busy.clear()

    def _process(self, job: _ObserveJob) -> None:
        self._on_debug("observing scene...")
        objects = call_with_retry(lambda: vision.describe_scene(job.frame), on_debug=self._on_debug)
        changed = self._memory.remember(objects) if self._memory is not None else bool(objects)
        with self._result_lock:
            self._new_observation = (objects, changed)
        self._on_debug(f"  saw {len(objects)} object(s), changed={changed}: {[o.label for o in objects]}")
