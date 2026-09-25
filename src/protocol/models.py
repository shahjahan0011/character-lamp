"""Shared message schema between the `character` service (the mind) and the
`body` service (the simulated lamp).

This is the one boundary a vision/language model is allowed to reach across.
It never sees a joint name or a joint angle -- only these closed, named
actions. The `body` service is the only code that knows how to turn a named
action into a joint trajectory. Keeping the schema in one file means the same
types work whether the transport underneath is an in-process asyncio.Queue
(what we use here) or a real network socket later -- see README for why.
"""

from __future__ import annotations

import time
from typing import Literal, Optional

from pydantic import BaseModel, Field

ActionKind = Literal[
    "look_at",      # params: pan (rad, +left), tilt (rad, +up)
    "point_at",     # params: pan (rad), tilt (rad) -- aim head/arm at a direction
    "nod",          # params: {} -- short acknowledgement gesture
    "shake_head",   # params: {} -- short "no" gesture
    "excited",      # params: {} -- enthusiastic bounce + brief bright flash
    "curious",      # params: {} -- head tilt, held briefly
    "think",        # params: {} -- slow pondering dip, repeated while waiting on a reply
    "set_light",    # params: on (bool), color ([r,g,b] 0-1), brightness (0-1)
    "play_sound",   # params: name (str) -- file under assets/sfx or assets/music
    "music_on",     # params: {} -- start the idle background music loop
    "music_off",    # params: {} -- stop the idle background music loop
    "speak",        # params: text (str) -- synthesized and played through speaker
    "observe",      # params: {} -- character service should capture+analyze a frame
    "idle_sway",    # params: {} -- small idle motion while waiting, signals "alive"
    "home",         # params: {} -- return to neutral resting pose
]


class Action(BaseModel):
    kind: ActionKind
    params: dict = Field(default_factory=dict)


class ActionBatch(BaseModel):
    """A planner (LLM/VLM) emits one of these per decision. `reason` is not
    executed -- kept only for logging/explainability in the demo and note."""

    actions: list[Action]
    reason: Optional[str] = None


TelemetryKind = Literal[
    "action_started",
    "action_done",
    "action_failed",
    "joint_state",
]


class Telemetry(BaseModel):
    kind: TelemetryKind
    payload: dict = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
