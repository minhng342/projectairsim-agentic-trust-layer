"""How a run finds its mission, scenario and scene (atal/core/paths.py): bundled files
through importlib.resources, your own files relative to where you run, read-only bundled
files, and errors that say what exists."""
import pytest

from atal.core.mission import PlanFileError, load_mission
from atal.core.paths import (
    ConfigNotFound,
    bundled_names,
    check_scene,
    config_root,
    read_config,
    runs_dir,
)
from atal.core.scenario import load_scenario
from atal.runtime.simulator import find_scene


def test_the_runner_defaults_are_bundled_and_load_from_any_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                                  # not the checkout
    m = load_mission("bundled:missions/sar/mission_dev.yaml")
    load_scenario("bundled:missions/sar/scenario_swarm_dev.yaml", m)
    assert "bundled:missions/sar/mission.yaml" in bundled_names("missions", ".yaml")


def test_your_own_file_is_found_relative_to_where_you_run(tmp_path, monkeypatch):
    own = tmp_path / "plans" / "mine.yaml"
    own.parent.mkdir()
    own.write_bytes(read_config("bundled:missions/sar/mission_dev.yaml")[1])
    monkeypatch.chdir(tmp_path)
    assert load_mission("plans/mine.yaml").mission_id


def test_a_missing_file_is_an_error_that_names_it_and_the_bundled_ones(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # an old checkout-relative spelling is not looked up anywhere else
    with pytest.raises(PlanFileError) as err:
        load_mission("missions/sar/mission_dev.yaml")
    text = str(err.value)
    assert "missions/sar/mission_dev.yaml: no such file" in text and str(tmp_path) in text
    assert "bundled:missions/sar/mission_dev.yaml" in text
    with pytest.raises(PlanFileError, match="bundled:missions/sar/nope.yaml: no such file"):
        load_mission("bundled:missions/sar/nope.yaml")


def test_bundled_scenes_resolve_with_every_robot_file_they_refer_to():
    names = bundled_names("sim_config", ".jsonc")
    scenes = [n for n in names if n.split("/")[-1].startswith("scene_")]
    assert "bundled:sim_config/scene_sar_three_drones.jsonc" in scenes
    for s in scenes:
        folder, name = find_scene(s)
        assert (folder / name).is_file() and check_scene(folder, name) == []


def test_a_scene_of_your_own_needs_its_robot_files_next_to_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    text = read_config("bundled:sim_config/scene_sar_one_drone.jsonc", "sim_config")[1]
    (tmp_path / "my_scene.jsonc").write_bytes(text)
    with pytest.raises(ConfigNotFound, match="robot_quadrotor_fastphysics.jsonc, not found"):
        find_scene("my_scene.jsonc")
    (tmp_path / "robot_quadrotor_fastphysics.jsonc").write_bytes(
        read_config("bundled:sim_config/robot_quadrotor_fastphysics.jsonc", "sim_config")[1])
    assert find_scene("my_scene.jsonc") == (tmp_path.resolve(), "my_scene.jsonc")
    with pytest.raises(ConfigNotFound, match="bundled:sim_config/scene_"):
        find_scene("nope.jsonc")


def test_run_output_defaults_to_runs_in_the_current_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runs_dir() == tmp_path / "runs"


def test_nothing_in_the_suite_wrote_into_the_bundled_files():
    names = {p.name for p in _files(config_root())}
    allowed = (".yaml", ".jsonc")
    assert names and all(n.endswith(allowed) for n in names), sorted(names)


def _files(node):
    for child in node.iterdir():
        if child.is_dir():
            if child.name != "__pycache__":
                yield from _files(child)
        else:
            yield child
