"""Cached geometry must produce the official observations and seeded trajectories."""

import numpy as np

from crashlearn import map_cache
from crashlearn.env import CrashLearnEnv
from crashlearn.map_cache import cached_map_loader


def trace():
    env = CrashLearnEnv(maps="Austin")
    result = []
    try:
        for track in ("Austin", "BrandsHatch", "Austin"):
            obs, _ = env.reset(seed=7, options={"map": track})
            result.append(obs)
            for _ in range(30):
                obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.1]))
                result.append(np.r_[obs, reward, terminated, truncated, info["progress_delta"]])
    finally:
        env.close()
    return result


def test_cached_maps_keep_the_official_seeded_trajectory(monkeypatch):
    from f110_gym.envs.laser_models import ScanSimulator2D

    expected = trace()
    monkeypatch.setattr(ScanSimulator2D, "set_map", cached_map_loader(ScanSimulator2D.set_map))
    actual = trace()
    for a, b in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(a, b)


def test_installing_twice_wraps_the_loader_once(monkeypatch):
    from f110_gym.envs.laser_models import ScanSimulator2D

    monkeypatch.setattr(ScanSimulator2D, "set_map", ScanSimulator2D.set_map)
    map_cache.install()
    installed = ScanSimulator2D.set_map
    map_cache.install()
    assert ScanSimulator2D.set_map is installed and installed.cached
