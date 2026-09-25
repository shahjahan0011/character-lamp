"""Ties perception signals to body actions.

For now this only handles engagement (demo moments 1 and 2): notice someone,
acknowledge them, dim back down when they leave. Speech/memory/goal states
get added here later without changing how this part works -- it just reads
whatever `EngagementWatcher.get_state()` says and reacts.

Deliberately not calling the LLM for this reaction (see design discussion):
it needs to feel instantaneous, and a network round-trip would undercut
that, so the acknowledgment is a fixed, hardcoded Action sequence.
"""

from __future__ import annotations

import time

from src.body.executor import ActionExecutor
from src.perception.engagement import EngagementWatcher
from src.protocol.models import Action

MAX_PAN_RAD = 0.7  # radians; matches base_yaw_joint's usable range for a look


class CharacterOrchestrator:
    def __init__(self, executor: ActionExecutor, watcher: EngagementWatcher):
        self.executor = executor
        self.watcher = watcher
        self._last_engaged = False

    def run_forever(self, poll_hz: float = 10.0) -> None:
        period = 1.0 / poll_hz
        try:
            while True:
                self.tick()
                time.sleep(period)
        except KeyboardInterrupt:
            pass

    def tick(self) -> None:
        """Check the current engagement state once and react to a
        just-happened transition. Exposed separately from run_forever so
        tests (and later, a bigger orchestrator loop covering speech/goals)
        can call it directly instead of only via a blocking loop."""
        state = self.watcher.get_state()
        if state.engaged and not self._last_engaged:
            self._on_engage(state.face_x_frac)
        elif not state.engaged and self._last_engaged:
            self._on_disengage()
        self._last_engaged = state.engaged

    def _on_engage(self, face_x_frac: float) -> None:
        pan = -face_x_frac * MAX_PAN_RAD
        self.executor.run(Action(kind="look_at", params={"pan": pan, "tilt": -0.1}))
        self.executor.run(Action(kind="nod", params={}))
        self.executor.run(
            Action(
                kind="set_light",
                params={"on": True, "color": [1.0, 0.95, 0.76], "brightness": 0.8},
            )
        )

    def _on_disengage(self) -> None:
        self.executor.run(
            Action(
                kind="set_light",
                params={"on": True, "color": [1.0, 0.95, 0.76], "brightness": 0.2},
            )
        )
        self.executor.run(Action(kind="home", params={}))
