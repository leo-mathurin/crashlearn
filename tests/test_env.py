"""Gymnasium wrapper contract (E-14). Skipped until crashlearn.env exists.

Module path: `crashlearn.env` = existing package + file name env.py imposed by the subject.
Assumptions: the module defines exactly one gymnasium.Env subclass, whose constructor
accepts `num_cars` (1..4). The raw simulator has no `done`: the wrapper decides.
"""

import gymnasium as gym
import numpy as np
import pytest
from conftest import import_if_present
from gymnasium.utils.env_checker import check_env, data_equivalence

env_module = import_if_present("crashlearn.env")

ENV_CLASSES = [
    obj
    for obj in vars(env_module).values()
    if isinstance(obj, type) and issubclass(obj, gym.Env) and obj.__module__ == env_module.__name__
]


@pytest.fixture
def env_class():
    assert len(ENV_CLASSES) == 1, f"expected one gymnasium.Env in crashlearn.env, got {ENV_CLASSES}"
    return ENV_CLASSES[0]


def rollout(env_class, steps=20):
    # the simulator is a process-wide singleton: never keep two envs alive at once
    env = env_class(num_cars=1)
    try:
        env.action_space.seed(0)
        obs, _ = env.reset(seed=0)
        trajectory = [obs]
        for _ in range(steps):
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            trajectory.append((obs, reward, terminated, truncated))
            if terminated or truncated:
                break
        return trajectory
    finally:
        env.close()


def test_gymnasium_check_env(env_class):
    env = env_class(num_cars=1)
    try:
        check_env(env)
    finally:
        env.close()


@pytest.mark.parametrize("num_cars", [1, 2, 3, 4])
def test_reset_then_step(env_class, num_cars):
    env = env_class(num_cars=num_cars)
    try:
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert type(terminated) is bool and type(truncated) is bool
    finally:
        env.close()


def test_seeded_short_episode_is_reproducible(env_class):
    assert data_equivalence(rollout(env_class), rollout(env_class))


def test_training_speed_limit_is_applied_to_controls(env_class):
    env = env_class(maps="IMS", max_speed=5.0)
    try:
        env.reset(seed=0)
        env.step(np.ones(2, dtype=np.float32))
        steer, speed = env_module.sim._sim_instance._actions[0]
        assert speed <= 5.0
        assert speed == pytest.approx(5.0)
        assert steer == pytest.approx(env_module.STEER_MAX)
        assert env.action_space.high[0] == pytest.approx(1 / 6)
    finally:
        env.close()


@pytest.mark.parametrize("speed", [0, -1, 11, float("nan"), float("inf")])
def test_invalid_training_speed_limit_is_refused(env_class, speed):
    with pytest.raises(ValueError, match="max_speed"):
        env_class(max_speed=speed)


def test_random_starts_are_seeded_and_restore_the_official_grid(env_class):
    env = env_class(maps="IMS", random_start=True)
    try:
        env_module.sim.set_map("IMS")
        original = env_module.sim._MAP_SX, env_module.sim._MAP_SY, env_module.sim._MAP_STHETA
        first, info = env.reset(seed=0)
        position = env_module.sim._get_sim()._sim.agents[0].state[:3].copy()
        assert info["raw_obs"]["lap_count"] == 0
        assert info["raw_obs"]["progress"] == pytest.approx(0)
        assert (
            env_module.sim._MAP_SX,
            env_module.sim._MAP_SY,
            env_module.sim._MAP_STHETA,
        ) == original
        repeated, _ = env.reset(seed=0)
        np.testing.assert_array_equal(first, repeated)
        env.reset(seed=1)
        assert np.linalg.norm(env_module.sim._get_sim()._sim.agents[0].state[:2] - position[:2]) > 1
    finally:
        env.close()
    official = env_class(maps="IMS")
    try:
        official.reset(seed=0)
        expected = env_module.sim._get_sim()._start_poses(1)[0]
        np.testing.assert_allclose(
            env_module.sim._get_sim()._sim.agents[0].state[[0, 1, 4]], expected
        )
    finally:
        official.close()


def test_a_race_seed_gives_the_engine_seed_used_without_gymnasium(env_class, monkeypatch):
    """confirm_submission.py replays races outside the env with `engine_seed`."""
    from crashlearn import engine_seed

    seeds = []
    reset = env_module.sim.reset
    monkeypatch.setattr(
        env_module.sim, "reset", lambda *a, **k: seeds.append(k["seed"]) or reset(*a, **k)
    )
    env = env_class(maps="IMS")
    try:
        env.reset(seed=104, options={"map": "IMS"})
    finally:
        env.close()
    assert seeds == [engine_seed(np.random.default_rng(104))]
