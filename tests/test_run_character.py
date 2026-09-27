"""scripts/run_character.py: argument parsing and the offline/mock demo.

The offline demo is the one place this suite exercises PyBullet for real
(headless, kinematic-only -- no GUI, no real time.sleep-bound physics) --
everything else about it (vision, camera, mic, network) is mocked/scripted.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import run_character


def test_parse_args_defaults():
    args = run_character.parse_args([])
    assert args.camera_index == 0
    assert args.headless is False
    assert args.offline is False
    assert args.force_engaged is False
    assert args.data_dir == "var"
    assert args.log_level == "INFO"


def test_parse_args_offline_and_headless():
    args = run_character.parse_args(["--offline", "--headless", "--data-dir", "/tmp/x"])
    assert args.offline is True
    assert args.headless is True
    assert args.data_dir == "/tmp/x"


def test_mock_model_alias_sets_offline():
    args = run_character.parse_args(["--mock-model"])
    assert args.offline is True


def test_offline_demo_runs_end_to_end_headless(tmp_path):
    """The actual deterministic mock demo, run for real (headless, no
    GUI) -- exercises CharacterOrchestrator/ToolGateway/GoalCoordinator/
    SceneMemory/ActionExecutor with only the vision network call
    scripted. Raises (via assert) on any workflow-invariant violation."""
    run_character.run_offline_demo(headless=True, data_dir=tmp_path)
    assert (tmp_path / "scene_memory.json").exists()
