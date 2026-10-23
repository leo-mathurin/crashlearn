"""Canonical features (E-16): one schema, fixed size, masked opponents, tolerant reading."""

import ast
import importlib.util
import math

import env_simulation as es
import numpy as np
import pytest
from agent_loader import create_dummy_obs
from conftest import SUBMISSION_DIRS

from crashlearn import features
from crashlearn.features import (
    EGO_SIZE,
    FEATURE_SIZE,
    MAX_OPPONENTS,
    OPPONENT_SIZE,
    adapt_observation,
    build_features,
)

TRAIN_AGENT_ATOL = 1e-6
EMPTY = {"x_rel": 0.0, "y_rel": 0.0, "yaw_rel": 0.0, "speed": 0.0, "progress": 0.0}


def slot(x=0.0, y=0.0, yaw=0.0, speed=0.0, progress=0.0, laps=0, status=1):
    return {
        "x_rel": x,
        "y_rel": y,
        "yaw_rel": yaw,
        "speed": speed,
        "progress": progress,
        "lap_count": laps,
        "status": status,
    }


def observation(*slots, progress=0.5, laps=1):
    obs = create_dummy_obs()
    obs.update(progress=progress, lap_count=laps)
    obs["opponents"] = dict(enumerate(slots))
    return obs


def opponent_rows(vec):
    return vec[EGO_SIZE:].reshape(MAX_OPPONENTS, OPPONENT_SIZE)


EGO = slot(status=0)
AHEAD = slot(x=3.0, y=-1.5, yaw=0.5, speed=6.0, progress=0.6, laps=1)


def test_constants_match_the_simulator():
    assert features.LIDAR_RAYS == es._LIDAR_RAYS
    assert features.LIDAR_MAX == es._LIDAR_MAX
    assert features.STEER_MAX == pytest.approx(es._PARAMS["s_max"])
    assert features.REQUIRED_LAPS == es._REQUIRED_LAPS
    assert features.MAX_OPPONENTS == es._MAX_SLOTS - 1


def test_the_simulator_schema_is_the_one_we_read(sim):
    """Loader, get_space_info and get_obs share one opponent schema since the 18/09 archive."""
    keys = set(slot())
    assert set(create_dummy_obs()["opponents"][0]) == keys
    assert all(set(o) == keys for o in sim.get_obs(0)["opponents"].values())


def test_features_module_needs_only_numpy():
    """It is copied into agent.py, where only NumPy and ONNX Runtime are allowed."""
    tree = ast.parse(open(features.__file__).read())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module)])
    }
    assert imported <= {"math", "numpy"}


def test_shape_dtype_bounds_and_determinism_on_real_observations(sim):
    for car in (0, 1):
        obs = sim.get_obs(car)
        vec = build_features(obs)
        assert vec.shape == (FEATURE_SIZE,) and vec.dtype == np.float32
        assert np.all(np.isfinite(vec)) and np.all(np.abs(vec) <= 1)
        np.testing.assert_array_equal(vec, build_features(obs))
        assert opponent_rows(vec)[0, 0] == 1.0  # the other car of the 2-car fixture
        assert not opponent_rows(vec)[1:].any()


def test_nominal_opponent_encoding():
    vec = build_features(observation(EGO, AHEAD))
    row = opponent_rows(vec)[0]
    np.testing.assert_allclose(
        row,
        [1.0, 3 / 15, -1.5 / 15, math.sin(0.5), math.cos(0.5), 0.6, 0.1],
        atol=1e-6,
    )
    assert not opponent_rows(vec)[1:].any()


def test_solo_and_empty_slots_are_all_zero():
    for obs in (create_dummy_obs(), observation(EGO, slot(status=0), slot(status=2))):
        assert not build_features(obs)[EGO_SIZE:].any()


@pytest.mark.parametrize("status", [0, 2], ids=["dnf_or_absent", "finished"])
def test_values_of_inactive_slots_are_ignored(status):
    garbage = slot(x=9.0, y=9.0, yaw=1.0, speed=7.0, progress=0.9, laps=2, status=status)
    np.testing.assert_array_equal(
        build_features(observation(EGO, AHEAD)),
        build_features(observation(EGO, AHEAD, garbage)),
    )


def test_opponents_are_sorted_nearest_first_whatever_their_slot():
    far, near = slot(x=10.0, speed=1.0), slot(x=2.0, speed=2.0)
    a = build_features(observation(EGO, far, near))
    b = build_features(observation(near, EGO, far))
    np.testing.assert_array_equal(a, b)
    assert opponent_rows(a)[0, 1] == pytest.approx(2 / 15)


def test_race_gap_tells_ahead_from_behind_and_saturates():
    behind = slot(x=-2.0, progress=0.4, laps=1)
    lapped = slot(x=5.0, progress=0.5, laps=0)
    rows = opponent_rows(build_features(observation(EGO, behind, lapped)))
    assert rows[0, 6] == pytest.approx(-0.1)  # nearest: 0.1 lap behind
    assert rows[1, 6] == -1.0  # a full lap behind


@pytest.mark.parametrize(
    "broken",
    [
        {},
        {"opponents": None},
        {"opponents": {0: None, 1: "x", 2: {"status": 1}}},
        {"lidar": None, "velocity": None, "steering": "fast"},
        {"lidar": np.full(30, 4.0), "progress": float("nan"), "lap_count": float("inf")},
        {"lidar": np.full(100, np.nan)},
        {"opponents": {1: {**EMPTY, "x_rel": float("nan"), "status": 1}}},
    ],
)
def test_incomplete_or_invalid_data_gives_neutral_finite_features(broken):
    obs = {**create_dummy_obs(), **broken}
    vec = build_features(obs)
    assert vec.shape == (FEATURE_SIZE,) and np.all(np.isfinite(vec))
    assert np.all(np.abs(vec) <= 1)


def test_missing_values_read_as_neutral():
    c = adapt_observation({"opponents": {1: {"status": 1}}, "lidar": [2.0] * 10})
    assert c["mask"].tolist() == [True, False, False]
    assert not c["opponents"].any()  # an active car with no data: present, at the ego
    np.testing.assert_array_equal(c["lidar"][:10], 2.0)
    np.testing.assert_array_equal(c["lidar"][10:], features.LIDAR_MAX)  # missing rays: free
    assert (c["velocity"], c["steering"], c["progress"]) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("submission_dir", SUBMISSION_DIRS, ids=lambda p: p.name)
def test_training_and_agent_features_match(submission_dir, sim):
    spec = importlib.util.spec_from_file_location(
        f"features_parity_{submission_dir.name}", submission_dir / "agent.py"
    )
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    for obs in (sim.get_obs(0), sim.get_obs(1), create_dummy_obs(), observation(EGO, AHEAD)):
        np.testing.assert_allclose(
            agent_module.build_features(obs), build_features(obs), atol=TRAIN_AGENT_ATOL, rtol=0
        )
