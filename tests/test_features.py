"""Canonical feature source contract (E-14). Skipped until crashlearn.features exists.

Naming assumptions (NOT fixed by the subject, rename here if E-14 decides otherwise):
- module `crashlearn.features`;
- `adapt_observation(obs)` maps both opponent schemas (simulator
  `x_rel/y_rel/yaw_rel/speed/progress/lap_count/status` and official loader
  `rel_x/rel_y/rel_dist/rel_yaw/velocity/active`) to one canonical form;
- `build_features(obs, info) -> np.ndarray`, normalized into [-FEATURE_BOUND, FEATURE_BOUND];
- submission agent.py files expose the same `build_features` at module level.
"""

import importlib.util
import math

import numpy as np
import pytest
from agent_loader import create_dummy_obs
from conftest import SUBMISSION_DIRS, import_if_present
from gymnasium.utils.env_checker import data_equivalence

features = import_if_present("crashlearn.features")
adapt_observation = getattr(features, "adapt_observation", None)
build_features = getattr(features, "build_features", None)

FEATURE_BOUND = 1.0
TRAIN_AGENT_ATOL = 1e-6

needs_adapter = pytest.mark.skipif(
    adapt_observation is None, reason="crashlearn.features.adapt_observation not defined"
)
needs_builder = pytest.mark.skipif(
    build_features is None, reason="crashlearn.features.build_features not defined"
)

NEUTRAL_REAL = {
    "x_rel": 0.0,
    "y_rel": 0.0,
    "yaw_rel": 0.0,
    "speed": 0.0,
    "progress": 0.0,
    "lap_count": 0,
    "status": 0,
}


def real_slot(x, y, yaw, speed, status):
    # progress/lap_count stay neutral: the loader schema has no equivalent field
    return {
        **NEUTRAL_REAL,
        "x_rel": x,
        "y_rel": y,
        "yaw_rel": yaw,
        "speed": speed,
        "status": status,
    }


def loader_slot(x, y, yaw, speed, status):
    return {
        "rel_x": x,
        "rel_y": y,
        "rel_dist": math.hypot(x, y),
        "rel_yaw": yaw,
        "velocity": speed,
        "active": int(status == 1),
    }


def observation(make_slot, slots, lidar=5.0):
    base = create_dummy_obs()
    base["lidar"] = np.full(100, lidar, dtype=np.float32)
    base["opponents"] = {i: make_slot(*values) for i, values in enumerate(slots)}
    return base


SITUATION = [(0, 0, 0, 0, 0), (2.0, -0.5, 0.3, 4.0, 1), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)]


@pytest.fixture
def observations(sim):
    real = sim.get_obs(0)
    return {
        "sim_car0": real,
        "sim_car1": sim.get_obs(1),
        "lidar_min": {**real, "lidar": np.full(100, 0.1, dtype=np.float32)},
        "lidar_max": {**real, "lidar": np.full(100, 15.0, dtype=np.float32)},
        "no_opponent": {**real, "opponents": {i: dict(NEUTRAL_REAL) for i in range(4)}},
        "loader_schema": create_dummy_obs(),
    }


@needs_adapter
def test_adapter_gives_same_result_for_both_schemas():
    assert data_equivalence(
        adapt_observation(observation(real_slot, SITUATION)),
        adapt_observation(observation(loader_slot, SITUATION)),
    )


@needs_adapter
@pytest.mark.parametrize("make_slot", [real_slot, loader_slot])
@pytest.mark.parametrize("slot", [0, 2], ids=["ego", "absent"])
def test_adapter_ignores_values_of_inactive_slots(make_slot, slot):
    noisy = list(SITUATION)
    noisy[slot] = (9.0, 9.0, 1.0, 7.0, 0)  # garbage values, still inactive
    assert data_equivalence(
        adapt_observation(observation(make_slot, SITUATION)),
        adapt_observation(observation(make_slot, noisy)),
    )


@needs_adapter
@pytest.mark.parametrize("make_slot", [real_slot, loader_slot])
def test_adapter_mask_reflects_activity(make_slot):
    inactive = list(SITUATION)
    inactive[1] = (*SITUATION[1][:4], 0)
    assert not data_equivalence(
        adapt_observation(observation(make_slot, SITUATION)),
        adapt_observation(observation(make_slot, inactive)),
    )


@needs_builder
def test_features_shape_dtype_bounds_and_determinism(sim, observations):
    info = sim.get_step_info()
    shapes = set()
    for name, obs in observations.items():
        first = build_features(obs, info)
        assert isinstance(first, np.ndarray) and first.ndim == 1, name
        assert first.dtype == np.float32, name
        assert np.all(np.isfinite(first)), name
        assert np.all(np.abs(first) <= FEATURE_BOUND), name
        np.testing.assert_array_equal(first, build_features(obs, info), err_msg=name)
        shapes.add(first.shape)
    assert len(shapes) == 1, f"feature size depends on the input: {shapes}"


@needs_builder
@pytest.mark.parametrize("submission_dir", SUBMISSION_DIRS, ids=lambda p: p.name)
def test_training_and_agent_features_match(submission_dir, sim, observations):
    spec = importlib.util.spec_from_file_location(
        f"features_parity_{submission_dir.name}", submission_dir / "agent.py"
    )
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    info = sim.get_step_info()
    for name, obs in observations.items():
        np.testing.assert_allclose(
            agent_module.build_features(obs, info),
            build_features(obs, info),
            atol=TRAIN_AGENT_ATOL,
            rtol=0,
            err_msg=name,
        )
