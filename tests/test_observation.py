"""Observation/DetectedObject schema validation and coordinate bounds."""

import pytest
from pydantic import ValidationError

from src.protocol.observation import DetectedObject, Observation


def test_detected_object_accepts_valid_bounds():
    obj = DetectedObject(label="Red mug", attributes="ceramic", image_x=0.5, image_y=0.5, confidence=0.9)
    assert obj.label == "Red mug"


@pytest.mark.parametrize("field_name,value", [("image_x", -0.1), ("image_x", 1.1), ("image_y", -0.1), ("image_y", 1.1)])
def test_detected_object_rejects_out_of_range_coordinates(field_name, value):
    kwargs = {"label": "x", "attributes": "", "image_x": 0.5, "image_y": 0.5, "confidence": 0.5}
    kwargs[field_name] = value
    with pytest.raises(ValidationError):
        DetectedObject(**kwargs)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_detected_object_rejects_out_of_range_confidence(value):
    with pytest.raises(ValidationError):
        DetectedObject(label="x", attributes="", image_x=0.5, image_y=0.5, confidence=value)


def test_detected_object_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        DetectedObject(
            label="x", attributes="", image_x=0.5, image_y=0.5, confidence=0.5, joint_angle=1.0
        )


def test_detected_object_rejects_empty_label():
    with pytest.raises(ValidationError):
        DetectedObject(label="", attributes="", image_x=0.5, image_y=0.5, confidence=0.5)


def test_observation_requires_at_least_one_pixel_dimension():
    with pytest.raises(ValidationError):
        Observation(purpose="scene", width=0, height=480, objects=[])


def test_observation_age_s_uses_monotonic_time():
    obs = Observation(purpose="scene", width=640, height=480, objects=[], captured_at_monotonic=100.0)
    assert obs.age_s(now_monotonic=100.0) == 0.0
    assert obs.age_s(now_monotonic=105.0) == 5.0
    # Never negative even if "now" is earlier than capture (clock skew guard).
    assert obs.age_s(now_monotonic=95.0) == 0.0


def test_observation_rejects_unknown_purpose():
    with pytest.raises(ValidationError):
        Observation(purpose="not_a_real_purpose", width=640, height=480, objects=[])
