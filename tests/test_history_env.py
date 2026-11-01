"""HistoryWrapper (E-18): the training stacks are the ones agent.py builds at the tournament."""

import env_simulation as es
import numpy as np
import pytest

from crashlearn.env import OBS_SIZE, CrashLearnEnv, controls_to_action
from crashlearn.features import build_features
from crashlearn.history import CarHistory
from crashlearn.history_env import HistoryWrapper

ACTION = controls_to_action(3.0, 0.1)


@pytest.fixture(scope="module", autouse=True)
def _restore_shared_sim_state():
    """Closing an env destroys the singleton: give later modules the state of `conftest.sim`."""
    yield
    es.close()
    es.set_map("example")
    es.reset(num_cars=2)
    es.simulation_step()


@pytest.fixture
def stacked():
    env = HistoryWrapper(CrashLearnEnv(maps="example"))
    yield env
    env.close()


def frames(stack: np.ndarray) -> np.ndarray:
    return stack.reshape(-1, OBS_SIZE)


def test_observation_space_covers_the_four_frames(stacked):
    obs, _ = stacked.reset(seed=0)
    assert stacked.observation_space.shape == (504,)
    assert stacked.observation_space.contains(obs)


def test_the_first_frame_fills_the_stack(stacked):
    obs, info = stacked.reset(seed=0)
    expected = build_features(info["raw_obs"])
    assert np.array_equal(frames(obs), np.tile(expected, (4, 1)))


def test_frames_shift_oldest_first(stacked):
    previous, _ = stacked.reset(seed=0)
    for _ in range(6):
        obs, _, _, _, info = stacked.step(ACTION)
        assert np.array_equal(frames(obs)[:3], frames(previous)[1:])
        assert np.array_equal(frames(obs)[3], build_features(info["raw_obs"]))
        previous = obs


def test_a_new_episode_keeps_nothing_of_the_previous_one(stacked):
    stacked.reset(seed=0)
    for _ in range(6):
        stacked.step(ACTION)
    obs, info = stacked.reset(seed=1)
    assert np.array_equal(frames(obs), np.tile(build_features(info["raw_obs"]), (4, 1)))


def test_an_episode_ended_by_the_env_starts_over_too():
    env = HistoryWrapper(CrashLearnEnv(maps="example", max_episode_steps=3))
    try:
        env.reset(seed=0)
        truncated = False
        while not truncated:
            _, _, _, truncated, _ = env.step(ACTION)
        obs, _ = env.reset()
        assert np.array_equal(frames(obs)[0], frames(obs)[3])
    finally:
        env.close()


def test_depth_one_is_the_plain_reactive_observation():
    env = CrashLearnEnv(maps="example")
    try:
        wrapped = HistoryWrapper(env, depth=1)
        obs, _ = wrapped.reset(seed=3)
        plain, _ = env.reset(seed=3)
        assert wrapped.observation_space.shape == env.observation_space.shape
        assert np.array_equal(obs, plain)
        obs, *_ = wrapped.step(ACTION)
        assert obs.shape == (OBS_SIZE,)
    finally:
        env.close()


def test_training_stacks_equal_what_a_submission_rebuilds(stacked):
    agent_history = CarHistory()
    obs, info = stacked.reset(seed=0)
    probe = {"agent_id": 0, "lidar": np.full(100, 5.0), "velocity": 0.0}
    agent_history.push(probe, {"step_count": 0}, build_features(probe))
    for _ in range(8):
        raw = info["raw_obs"]
        rebuilt = agent_history.push(raw, info["step_info"], build_features(raw))
        assert np.array_equal(obs, rebuilt)
        obs, _, _, _, info = stacked.step(ACTION)
