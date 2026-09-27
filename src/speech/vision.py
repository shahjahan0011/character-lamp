"""Scene understanding via Gemini vision -- demo moment 4 (scene memory)
and the perception half of goal-directed action (demo moment 5).

One frame per call. Uses response_format's JSON-schema constraint
(confirmed live: interactions.create() supports
response_format={"type": "text", "mime_type": "application/json",
"schema": {...}}) rather than a free-text-plus-regex parser -- also
confirmed live, grounding normalized image_x/image_y this way against a
test image with known object positions landed within ~1% of the computed
true center, which a label/attribute-only parser had no way to produce at
all. Falls back to returning an empty object list (not a crash) if the
model's response fails schema validation, since a single bad observation
shouldn't be worse than "saw nothing this time."

Frames are never persisted here -- the caller (SceneObserver) passes a
frame already in memory and this module only ever sends bytes to Gemini,
never writes them to disk.
"""

from __future__ import annotations

import base64
import json
import time
import uuid

import cv2
import numpy as np
from pydantic import ValidationError

from src.protocol.observation import DetectedObject, Observation, ObservationPurpose

from .gemini_client import VISION_MODEL, collect_text_stream, get_client

DESCRIBE_PROMPT = (
    "Look at this image from a small desk lamp's webcam. Describe up to 6 "
    "notable things you can see in the surroundings -- furniture, objects "
    "on a desk or shelf, decor, anything a person might ask 'what's around "
    "you?' or 'what does it look like in here?' about, or might later ask "
    "you to point at. Skip people themselves and pure background (bare "
    "walls/floor with nothing on them).\n\n"
    "For each object give: a short label, 1-2 key visual attributes "
    "(shape, material, position), an optional single dominant color word, "
    "its approximate CENTER position as image_x/image_y normalized so "
    "0.0 is the left/top edge and 1.0 is the right/bottom edge, and your "
    "confidence 0.0-1.0 that the label and position are correct. If "
    "there is truly nothing describable in view, return an empty list."
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "attributes": {"type": "string"},
                    "color": {"type": "string"},
                    "image_x": {"type": "number"},
                    "image_y": {"type": "number"},
                    "confidence": {"type": "number"},
                },
                "required": ["label", "attributes", "image_x", "image_y", "confidence"],
            },
        }
    },
    "required": ["objects"],
}


def describe_scene(
    frame_bgr: np.ndarray, purpose: ObservationPurpose = "scene", timeout_s: float = 30.0
) -> Observation:
    height, width = frame_bgr.shape[:2]
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
        response_format={"type": "text", "mime_type": "application/json", "schema": _RESPONSE_SCHEMA},
        stream=True,
        timeout=timeout_s,
    )
    raw = collect_text_stream(stream)
    objects = _parse_objects(raw)
    return Observation(
        observation_id=uuid.uuid4().hex,
        captured_at_monotonic=time.monotonic(),
        purpose=purpose,
        width=width,
        height=height,
        objects=objects,
    )


def _parse_objects(raw: str) -> list[DetectedObject]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    objects = []
    for item in payload.get("objects", []):
        try:
            objects.append(DetectedObject.model_validate(item))
        except ValidationError:
            # Reject the one malformed/out-of-range entry, not the whole
            # observation -- one bad object shouldn't discard everything
            # else the model correctly saw in the same frame.
            continue
    return objects
