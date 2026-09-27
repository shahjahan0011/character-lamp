"""Keyed store of recent Observations, keyed by observation_id.

Freshness uses time.monotonic() throughout -- wall-clock time can jump
(NTP sync, sleep/wake, DST) in a way that would corrupt a freshness check;
monotonic time can't. Goal actions reject an observation_id this registry
doesn't know about, or one it knows about but considers stale (see
is_fresh()) -- this is what makes "a fresh goal_planning observation must
exist before scene-directed action" and "verification observation newer
than final action" enforceable rather than just a prompt-level request.
"""

from __future__ import annotations

import threading
from collections import deque

from src.protocol.observation import Observation, ObservationPurpose

DEFAULT_MAX_AGE_S = 30.0
DEFAULT_MAX_ENTRIES = 50


class ObservationRegistry:
    def __init__(self, max_age_s: float = DEFAULT_MAX_AGE_S, max_entries: int = DEFAULT_MAX_ENTRIES):
        self._max_age_s = max_age_s
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: dict[str, Observation] = {}
        self._order: deque[str] = deque()

    def add(self, observation: Observation) -> None:
        with self._lock:
            self._entries[observation.observation_id] = observation
            self._order.append(observation.observation_id)
            while len(self._order) > self._max_entries:
                oldest = self._order.popleft()
                self._entries.pop(oldest, None)

    def get(self, observation_id: str) -> Observation | None:
        with self._lock:
            return self._entries.get(observation_id)

    def is_fresh(self, observation_id: str, max_age_s: float | None = None) -> bool:
        obs = self.get(observation_id)
        if obs is None:
            return False
        limit = max_age_s if max_age_s is not None else self._max_age_s
        return obs.age_s() <= limit

    def latest(self, purpose: ObservationPurpose | None = None) -> Observation | None:
        with self._lock:
            for observation_id in reversed(self._order):
                obs = self._entries.get(observation_id)
                if obs is not None and (purpose is None or obs.purpose == purpose):
                    return obs
        return None

    def is_newer_than(self, observation_id: str, than_monotonic: float) -> bool:
        """True iff observation_id exists and was captured strictly after
        than_monotonic -- how GoalCoordinator enforces "the verification
        observation must be newer than the final action's completion"."""
        obs = self.get(observation_id)
        return obs is not None and obs.captured_at_monotonic > than_monotonic
