"""Runs the scene-description Gemini call (demo moment 4, and the
perception half of demo moment 5's goal workflow) on its own background
thread, so a several-second vision call never blocks
CharacterOrchestrator.tick().

Serves two callers with the same job type: CharacterOrchestrator's
periodic/once-per-engagement ambient "scene" reobservation (fire-and-
forget, silently updates SceneMemory), and ToolGateway's on-demand
request_observation tool calls (object_memory/goal_planning/
goal_verification -- tagged with a tool_call_id so the result can
complete that specific Live tool call once the real vision call finishes,
matching Behavior.BLOCKING: the model actually waits for this).

Used to also process spoken-utterance replies (this class was named
DialogueWorker) before conversation moved to a persistent Gemini Live
session (see live_client.py) -- that round trip doesn't exist as a
separate background job anymore, so this class now only ever does one
kind of job: describe a frame, remember what it saw (for "scene" only),
register the observation, and report completion.
"""

from __future__ import annotations

import queue
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.character.memory import SceneMemory
from src.character.observation_registry import ObservationRegistry
from src.protocol.observation import Observation, ObservationPurpose
from src.speech import vision
from src.speech.gemini_client import GeminiStreamError, call_with_retry

_RATE_LIMIT_CODES = frozenset({"rate_limit_exceeded", "quota_exceeded", "too_many_requests"})


@dataclass
class _ObserveJob:
    frame: np.ndarray
    purpose: ObservationPurpose
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass
class ObservationResult:
    observation: Observation
    changed: bool  # meaningful only for purpose="scene" -- SceneMemory dedup signal
    tool_call_id: str | None = None
    tool_name: str | None = None


class SceneObserver:
    def __init__(
        self,
        memory: SceneMemory | None = None,
        registry: ObservationRegistry | None = None,
        on_debug: Callable[[str], None] | None = None,
    ):
        self._memory = memory
        self._registry = registry
        self._on_debug = on_debug or (lambda msg: None)
        self._input: queue.Queue[_ObserveJob] = queue.Queue()
        self._busy = threading.Event()
        self._stop = threading.Event()

        self._result_lock = threading.Lock()
        self._new_observation: ObservationResult | None = None
        self._failed = False
        self._rate_limited = False
        self._failed_tool_call: tuple[str, str, str] | None = None  # (call_id, tool_name, message)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit_observation(
        self,
        frame: np.ndarray,
        purpose: ObservationPurpose = "scene",
        tool_call_id: str | None = None,
        tool_name: str | None = None,
    ) -> bool:
        """Queues a scene-description job. Refuses (and drops the frame)
        if a previous observation is still in flight -- the caller (either
        the periodic reobserve timer, or ToolGateway) is expected to treat
        False as "try again shortly" rather than blocking on it."""
        if self._busy.is_set():
            self._on_debug("scene observer: busy, skipping this observation")
            return False
        self._input.put(_ObserveJob(frame=frame, purpose=purpose, tool_call_id=tool_call_id, tool_name=tool_name))
        return True

    def is_busy(self) -> bool:
        return self._busy.is_set()

    def pop_new_observation(self) -> ObservationResult | None:
        """Non-blocking poll from the main thread."""
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
        the normal cadence and hitting the same wall again immediately."""
        with self._result_lock:
            rate_limited, self._rate_limited = self._rate_limited, False
        return rate_limited

    def pop_failed_tool_call(self) -> tuple[str, str, str] | None:
        """Non-blocking poll: if the failed observation was a tool-driven
        request_observation, returns (call_id, tool_name, message) so the
        caller can still submit a structured tool error and complete that
        Live turn rather than leaving it hanging (BLOCKING behavior means
        Gemini is actually waiting on this)."""
        with self._result_lock:
            failed, self._failed_tool_call = self._failed_tool_call, None
        return failed

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        self._input.put(None)  # unblock the worker thread's queue.get()
        self._thread.join(timeout=timeout_s)

    def _run(self) -> None:
        while not self._stop.is_set():
            job = self._input.get()
            if job is None:
                continue
            self._busy.set()
            try:
                self._process(job)
            except Exception as exc:  # noqa: BLE001 -- one bad observation shouldn't kill the worker
                self._on_debug(f"scene observer error:\n{traceback.format_exc()}")
                is_rate_limit = isinstance(exc, GeminiStreamError) and exc.code in _RATE_LIMIT_CODES
                with self._result_lock:
                    self._failed = True
                    self._rate_limited = is_rate_limit
                    if job.tool_call_id is not None:
                        self._failed_tool_call = (job.tool_call_id, job.tool_name or "request_observation", str(exc))
            finally:
                self._busy.clear()

    def _process(self, job: _ObserveJob) -> None:
        self._on_debug(f"observing scene (purpose={job.purpose})...")
        observation = call_with_retry(
            lambda: vision.describe_scene(job.frame, purpose=job.purpose), on_debug=self._on_debug
        )
        if self._registry is not None:
            self._registry.add(observation)
        changed = False
        if job.purpose == "scene" and self._memory is not None:
            changed = self._memory.remember(observation.objects, source_observation_id=observation.observation_id)
        with self._result_lock:
            self._new_observation = ObservationResult(
                observation=observation, changed=changed, tool_call_id=job.tool_call_id, tool_name=job.tool_name
            )
        self._on_debug(
            f"  saw {len(observation.objects)} object(s), changed={changed}: "
            f"{[o.label for o in observation.objects]}"
        )
