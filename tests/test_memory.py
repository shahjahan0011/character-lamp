"""SceneMemory: dedup, atomic persistence, corrupt-file backup, recall."""

import json

import pytest

from src.character.memory import SceneMemory
from src.protocol.observation import DetectedObject


def make_object(label="Red mug", color="red", attributes="ceramic", x=0.5, y=0.5, confidence=0.9):
    return DetectedObject(label=label, color=color, attributes=attributes, image_x=x, image_y=y, confidence=confidence)


def test_purely_in_memory_without_persist_path():
    memory = SceneMemory(persist_path=None)
    changed = memory.remember([make_object()], source_observation_id="obs1")
    assert changed is True
    assert len(memory) == 1


def test_dedup_by_label_and_color_does_not_append_duplicate():
    memory = SceneMemory(persist_path=None)
    memory.remember([make_object(label="Red mug", color="red")], source_observation_id="obs1")
    changed = memory.remember([make_object(label="Red mug", color="red")], source_observation_id="obs2")
    assert len(memory) == 1
    assert changed is False  # identical attributes/position -- nothing new to report


def test_same_label_different_color_is_a_separate_record():
    memory = SceneMemory(persist_path=None)
    memory.remember([make_object(label="Mug", color="red")], source_observation_id="obs1")
    memory.remember([make_object(label="Mug", color="blue")], source_observation_id="obs2")
    assert len(memory) == 2


def test_dedup_is_case_insensitive():
    memory = SceneMemory(persist_path=None)
    memory.remember([make_object(label="Red Mug", color="RED")], source_observation_id="obs1")
    memory.remember([make_object(label="red mug", color="red")], source_observation_id="obs2")
    assert len(memory) == 1


def test_changed_attributes_update_in_place_not_append():
    memory = SceneMemory(persist_path=None)
    memory.remember([make_object(attributes="ceramic")], source_observation_id="obs1")
    changed = memory.remember([make_object(attributes="ceramic, chipped")], source_observation_id="obs2")
    assert changed is True
    assert len(memory) == 1
    assert memory.all_records()[0].attributes == "ceramic, chipped"
    assert memory.all_records()[0].source_observation_id == "obs2"


def test_max_entries_enforced():
    memory = SceneMemory(max_entries=2, persist_path=None)
    for i in range(5):
        memory.remember([make_object(label=f"Object{i}", color=None)], source_observation_id=f"obs{i}")
    assert len(memory) == 2


def test_remember_one_tool_path():
    memory = SceneMemory(persist_path=None)
    record = memory.remember_one(
        label="Blue cup", color="blue", attributes="plastic", image_x=0.3, image_y=0.6,
        confidence=0.8, source_observation_id="obs1", notes=None,
    )
    assert record.label == "Blue cup"
    assert len(memory) == 1


def test_sanitizes_control_characters_and_truncates():
    memory = SceneMemory(persist_path=None)
    memory.remember_one(
        label="Mug\x00\x01", color=None, attributes="a" * 500, image_x=0.5, image_y=0.5,
        confidence=0.5, source_observation_id="obs1", notes=None,
    )
    record = memory.all_records()[0]
    assert "\x00" not in record.label
    assert len(record.attributes) <= 200


def test_recall_matches_by_token():
    memory = SceneMemory(persist_path=None)
    memory.remember([make_object(label="Red mug", color="red", attributes="ceramic")], source_observation_id="obs1")
    memory.remember([make_object(label="Blue cup", color="blue", attributes="plastic")], source_observation_id="obs2")
    results = memory.recall("mug")
    assert len(results) == 1
    assert results[0].label == "Red mug"


def test_recall_falls_back_to_recent_when_no_token_matches():
    memory = SceneMemory(persist_path=None)
    memory.remember([make_object(label="Red mug")], source_observation_id="obs1")
    results = memory.recall("completely unrelated gibberish query")
    assert len(results) == 1  # falls back to most-recent rather than returning nothing


def test_recall_respects_limit():
    memory = SceneMemory(persist_path=None)
    for i in range(5):
        memory.remember([make_object(label=f"Item{i}", color=None)], source_observation_id=f"obs{i}")
    assert len(memory.recall("item", limit=2)) == 2


def test_atomic_persistence_round_trips(tmp_path):
    path = tmp_path / "scene_memory.json"
    memory = SceneMemory(persist_path=path)
    memory.remember([make_object()], source_observation_id="obs1")
    assert path.exists()
    reloaded = SceneMemory(persist_path=path)
    assert len(reloaded) == 1
    assert reloaded.all_records()[0].label == "Red mug"


def test_no_tmp_file_left_behind_after_save(tmp_path):
    path = tmp_path / "scene_memory.json"
    memory = SceneMemory(persist_path=path)
    memory.remember([make_object()], source_observation_id="obs1")
    assert not (tmp_path / "scene_memory.json.tmp").exists()


def test_corrupt_file_is_backed_up_not_overwritten(tmp_path):
    path = tmp_path / "scene_memory.json"
    path.write_text("{not valid json!!!")
    debug_messages = []
    memory = SceneMemory(persist_path=path, on_debug=debug_messages.append)
    assert len(memory) == 0  # starts fresh rather than crashing
    backups = list(tmp_path.glob("scene_memory.json.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text() == "{not valid json!!!"
    assert any("corrupt" in m for m in debug_messages)


def test_malformed_record_in_file_is_treated_as_corrupt(tmp_path):
    path = tmp_path / "scene_memory.json"
    path.write_text(json.dumps({"records": [{"label": "x", "image_x": 5.0}]}))  # image_x out of bounds
    memory = SceneMemory(persist_path=path)
    assert len(memory) == 0
    assert list(tmp_path.glob("scene_memory.json.corrupt-*"))


def test_remember_rejects_malformed_direct_call():
    memory = SceneMemory(persist_path=None)
    with pytest.raises(ValueError):
        memory.remember_one(
            label="", color=None, attributes="", image_x=0.5, image_y=0.5,
            confidence=0.5, source_observation_id="obs1", notes=None,
        )
