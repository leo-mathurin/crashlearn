"""Gymnasium wrapper contract (E-13). Skipped until crashlearn.env exists.

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
