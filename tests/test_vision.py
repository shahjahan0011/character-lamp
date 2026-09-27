"""vision.py's structured-output parsing -- no real Gemini calls."""

import json
from unittest.mock import MagicMock, patch

import numpy as np

from src.speech import vision


def test_parse_objects_accepts_valid_payload():
    raw = json.dumps({
        "objects": [
            {"label": "Red mug", "attributes": "ceramic", "image_x": 0.5, "image_y": 0.5, "confidence": 0.9},
        ]
    })
    objects = vision._parse_objects(raw)
    assert len(objects) == 1
    assert objects[0].label == "Red mug"


def test_parse_objects_rejects_one_malformed_entry_but_keeps_others():
    raw = json.dumps({
        "objects": [
            {"label": "Good", "attributes": "", "image_x": 0.5, "image_y": 0.5, "confidence": 0.9},
            {"label": "Bad", "attributes": "", "image_x": 5.0, "image_y": 0.5, "confidence": 0.9},  # out of bounds
        ]
    })
    objects = vision._parse_objects(raw)
    assert len(objects) == 1
    assert objects[0].label == "Good"


def test_parse_objects_handles_invalid_json():
    assert vision._parse_objects("not json at all") == []


def test_parse_objects_handles_non_dict_json():
    assert vision._parse_objects(json.dumps([1, 2, 3])) == []


def test_parse_objects_handles_missing_objects_key():
    assert vision._parse_objects(json.dumps({"something_else": []})) == []


def test_describe_scene_returns_structured_observation_mocked():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    fake_response_json = json.dumps({
        "objects": [
            {"label": "Book", "attributes": "hardcover", "color": "red", "image_x": 0.2, "image_y": 0.3, "confidence": 0.8},
        ]
    })
    with patch("src.speech.vision.get_client") as mock_get_client, \
         patch("src.speech.vision.collect_text_stream", return_value=fake_response_json):
        mock_get_client.return_value.interactions.create.return_value = MagicMock()
        obs = vision.describe_scene(frame, purpose="goal_planning")

    assert obs.purpose == "goal_planning"
    assert obs.width == 320
    assert obs.height == 240
    assert len(obs.objects) == 1
    assert obs.objects[0].label == "Book"
    assert obs.objects[0].color == "red"
    assert obs.observation_id  # non-empty, unique per call
