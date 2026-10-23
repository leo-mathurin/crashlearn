"""Per-car temporal memory (E-13): separate histories, reset detection, dry-run, recorder."""

import ast
import json
import subprocess
import sys
import textwrap

import env_simulation as es
import numpy as np
import pytest
from agent_loader import create_dummy_info, create_dummy_obs
from conftest import ROOT

from crashlearn import history
from crashlearn.history import DEPTH, MISSING_ID, CarHistory, car_id, decision_step, is_probe


def obs_of(car, laps=0, lidar=7.0):
    obs = create_dummy_obs()
    obs.update(agent_id=car, lap_count=laps, velocity=1.0, lidar=np.full(100, lidar))
    return obs


def info_at(step):
    return {"step_count": step, "time_elapsed": step / 20}


def frame(value):
    return np.full(3, value, dtype=np.float32)


def frames_of(stack):
    return stack.reshape(DEPTH, -1)[:, 0].tolist()


def test_history_module_needs_only_numpy():
    """It is copied into agent.py, where only NumPy and ONNX Runtime are allowed."""
    tree = ast.parse(open(history.__file__).read())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module)])
    }
    assert imported <= {"math", "numpy"}


def test_constants_match_the_simulator():
    assert history.DECISION_FREQ_HZ == es._DECISION_FREQ_HZ
    dummy = create_dummy_obs()
    assert np.all(np.asarray(dummy["lidar"]) == history.PROBE_LIDAR)


def test_first_frame_fills_the_four_slots():
    stack = CarHistory().push(obs_of(0), info_at(1), frame(5))
    assert DEPTH == 4
    assert stack.shape == (4 * 3,) and stack.dtype == np.float32
    assert frames_of(stack) == [5, 5, 5, 5]


def test_frames_shift_oldest_first():
    h = CarHistory()
    for step in range(1, 7):
        stack = h.push(obs_of(0), info_at(step), frame(step))
    assert frames_of(stack) == [3, 4, 5, 6]


def test_alternating_cars_keep_separate_histories():
    h = CarHistory()
    for step in range(1, 6):
        for car in (2, 0, 3, 1):
            stack = h.push(obs_of(car), info_at(step), frame(10 * car + step))
            expected = [10 * car + max(1, s) for s in range(step - 3, step + 1)]
            assert frames_of(stack) == expected


def test_step_counter_going_back_starts_a_new_episode():
    h = CarHistory()
    for step in range(1, 6):
        h.push(obs_of(0), info_at(step), frame(step))
    assert frames_of(h.push(obs_of(0), info_at(1), frame(100))) == [100] * 4
    assert frames_of(h.push(obs_of(0), info_at(2), frame(101))) == [100, 100, 100, 101]


def test_back_to_back_short_episodes_are_told_apart():
    """An episode ending at step 1 and the next one starting at step 1: same counter."""
    h = CarHistory()
    h.push(obs_of(0), info_at(1), frame(1))
    assert frames_of(h.push(obs_of(0), info_at(1), frame(9))) == [9] * 4


def test_a_skipped_step_restarts_the_history():
    h = CarHistory()
    h.push(obs_of(0), info_at(1), frame(1))
    h.push(obs_of(0), info_at(2), frame(2))
    assert frames_of(h.push(obs_of(0), info_at(5), frame(5))) == [5] * 4


def test_time_elapsed_stands_in_for_a_missing_step_count():
    h = CarHistory()
    h.push(obs_of(0), {"time_elapsed": 0.25}, frame(1))
    assert frames_of(h.push(obs_of(0), {"time_elapsed": 0.30}, frame(2))) == [1, 1, 1, 2]
    assert frames_of(h.push(obs_of(0), {"time_elapsed": 0.05}, frame(3))) == [3] * 4


def test_lap_count_going_back_starts_a_new_episode_without_a_clock():
    h = CarHistory()
    h.push(obs_of(0, laps=2), {}, frame(1))
    assert frames_of(h.push(obs_of(0, laps=2), {}, frame(2))) == [1, 1, 1, 2]
    assert frames_of(h.push(obs_of(0, laps=0), {}, frame(3))) == [3] * 4


def test_loader_dry_run_leaves_no_trace():
    """The dummy frame (agent_id 0, step 0) comes right before the race's step 1."""
    h = CarHistory()
    dry = h.push(create_dummy_obs(), create_dummy_info(), frame(-1))
    assert frames_of(dry) == [-1] * 4
    assert frames_of(h.push(obs_of(0), info_at(1), frame(1))) == [1] * 4


def test_dry_run_without_agent_id_does_not_crash():
    obs = create_dummy_obs()
    del obs["agent_id"]
    assert frames_of(CarHistory().push(obs, create_dummy_info(), frame(0))) == [0] * 4


def test_missing_agent_id_is_kept_apart_from_car_zero():
    h = CarHistory()
    anonymous = obs_of(0)
    del anonymous["agent_id"]
    h.push(obs_of(0), info_at(1), frame(1))
    assert frames_of(h.push(anonymous, info_at(1), frame(50))) == [50] * 4
    assert frames_of(h.push(obs_of(0), info_at(2), frame(2))) == [1, 1, 1, 2]


@pytest.mark.parametrize("value", [None, "a", 1.5, -2, True, float("nan")])
def test_invalid_agent_ids_fall_back(value):
    assert car_id({"agent_id": value}) == MISSING_ID


@pytest.mark.parametrize(
    "obs, info",
    [(None, None), ({}, {}), ("obs", "info"), ({"lidar": "x"}, {"step_count": "x"})],
)
def test_malformed_inputs_never_raise(obs, info):
    h = CarHistory()
    h.push(obs, info, frame(1))
    assert frames_of(h.push(obs, info, frame(2))) == [1, 1, 1, 2]
    assert decision_step(info) is None
    assert not is_probe(obs, info)


def test_a_real_frame_at_step_zero_is_not_a_probe():
    """demo.py calls predict() on the real step-0 observation."""
    es.reset(1, seed=0)
    try:
        assert not is_probe(es.get_obs(0), es.get_step_info())
    finally:
        es.close()


def test_returned_stack_is_a_copy():
    h = CarHistory()
    stack = h.push(obs_of(0), info_at(1), frame(1))
    stack[:] = 0
    assert frames_of(h.push(obs_of(0), info_at(2), frame(2))) == [1, 1, 1, 2]


def test_feature_size_change_restarts_the_history():
    h = CarHistory()
    h.push(obs_of(0), info_at(1), frame(1))
    stack = h.push(obs_of(0), info_at(2), np.ones(5))
    assert stack.shape == (4 * 5,)


AGENT = """
import importlib.util
import json

import numpy as np

_spec = importlib.util.spec_from_file_location("e13_history", {history!r})
_history = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_history)


class Agent:
    def __init__(self):
        self.history = _history.CarHistory()

    def predict(self, obs, info):
        car = obs.get("agent_id", -1)
        features = np.concatenate([[car, obs["progress"]], np.asarray(obs["lidar"])[:2]])
        stack = self.history.push(obs, info, features)
        with open({log!r}, "a") as f:
            record = {{
                "instance": id(self),
                "car": car,
                "step": info.get("step_count"),
                "frame": np.asarray(features, dtype=np.float32).tolist(),
                "stack": stack.tolist(),
            }}
            f.write(json.dumps(record) + "\\n")
        return 4.0, 0.0
"""


# sim_recorder clones a single loaded submission across every car, but the 2026-09-18 loader
# refuses a flat submission for more than one car. Only the discovery is replaced here, so that
# it returns one team: the dry-run and the race loop are the recorder's own.
SELF_PLAY = """
import runpy
import sys

sys.path.insert(0, {simulation!r})
import agent_loader

agent_loader.discover_and_load_submissions = lambda path, max_teams=4: [
    agent_loader.validate_and_load_agent(path, team_name="e13", unique_id=0)
]
sys.argv = [{recorder!r}, *sys.argv[1:]]
runpy.run_path({recorder!r}, run_name="__main__")
"""


@pytest.mark.slow
def test_no_memory_mixing_in_the_recorder_sharing_one_instance(tmp_path):
    submission = tmp_path / "submission"
    submission.mkdir()
    log = tmp_path / "calls.jsonl"
    agent = AGENT.format(history=history.__file__, log=str(log))
    (submission / "agent.py").write_text(textwrap.dedent(agent))
    (submission / "model.onnx").write_bytes(b"")  # the loader only checks that it exists

    simulation = ROOT / "vendor/simulation"
    runner = tmp_path / "self_play.py"
    recorder = simulation / "sim_recorder.py"
    runner.write_text(SELF_PLAY.format(simulation=str(simulation), recorder=str(recorder)))
    command = [sys.executable, str(runner), "--submission", str(submission)]
    command += ["--num-cars", "4", "--steps", "40", "--seed", "0", "--map", "Budapest"]
    command += ["--out", str(tmp_path / "episode.npz")]
    run = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "replicated across 4 cars" in run.stdout

    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len({c["instance"] for c in calls}) == 1  # one Agent for the four cars
    dry_run, race = calls[0], calls[1:]
    assert dry_run["step"] == 0
    assert {c["car"] for c in race} == {0, 1, 2, 3}

    for car in range(4):
        own = [c for c in race if c["car"] == car]
        assert [c["step"] for c in own] == list(range(1, len(own) + 1))
        seen = [own[0]["frame"]] * DEPTH
        for c in own:
            seen = seen[1:] + [c["frame"]]
            np.testing.assert_array_equal(np.reshape(c["stack"], (DEPTH, -1)), seen)
