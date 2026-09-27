"""Scene understanding via Gemini vision -- demo moment 4 (scene memory).

One frame per call. Originally triggered once per engagement; now also
called periodically while engaged (see CharacterOrchestrator's re-
observation timer) so the lamp notices *changes* in its surroundings, not
just whatever was in view the instant someone first looked at it.

The prompt was originally scoped to "notable, non-furniture" objects only,
which in practice meant an ordinary desk/office scene often came back
empty (confirmed live: "saw 0 object(s): []" against real rooms) -- not
useful for "describe what's around you". Broadened to general surroundings
(furniture included), still capped at a handful of items so memory/context
stays compact.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass

import cv2
import numpy as np

from .gemini_client import VISION_MODEL, collect_text_stream, get_client

DESCRIBE_PROMPT = (
    "Look at this image from a small desk lamp's webcam. Describe up to 5 "
    "notable things you can see in the surroundings -- furniture, objects "
    "on a desk or shelf, decor, anything a person might ask 'what's around "
    "you?' or 'what does it look like in here?' about. Skip people "
    "themselves and pure background (bare walls/floor with nothing on "
    "them). For each, give a short label and 1-2 key visual attributes "
    "(color, shape, material, position like 'on the left'). Respond with "
    "ONE item per line, in EXACTLY this format, nothing else:\n"
    "LABEL: <short name> | ATTRIBUTES: <comma-separated attributes>\n"
    "If there is truly nothing describable in view, respond with exactly: "
    "NONE"
)

_OBJECT_RE = re.compile(r"LABEL:\s*(.+?)\s*\|\s*ATTRIBUTES:\s*(.+)", re.IGNORECASE)


@dataclass
class ObservedObject:
    label: str
    attributes: str


def describe_scene(frame_bgr: np.ndarray, timeout_s: float = 30.0) -> list[ObservedObject]:
    ok, jpeg = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise RuntimeError("Failed to JPEG-encode the frame")

    client = get_client()
    stream = client.interactions.create(
        model=VISION_MODEL,
        input=[
            {"type": "text", "text": DESCRIBE_PROMPT},
            {
                "type": "image",
                "data": base64.b64encode(jpeg.tobytes()).decode("utf-8"),
                "mime_type": "image/jpeg",
            },
        ],
        stream=True,
        timeout=timeout_s,
    )
    return _parse_objects(collect_text_stream(stream))


def _parse_objects(text: str) -> list[ObservedObject]:
    if text.strip().upper() == "NONE":
        return []
    objects = []
    for line in text.splitlines():
        match = _OBJECT_RE.search(line)
        if match:
            objects.append(ObservedObject(label=match.group(1).strip(), attributes=match.group(2).strip()))
    return objects
