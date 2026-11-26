"""CrashLearnEnv observations stacked by CarHistory (E-18, docs/results/e-18/README.md).

Kept out of `history.py`, which must stay pure NumPy for the submission `agent.py`. Same
frames as `agent.py` at the tournament: the first one fills the stack, unlike the zeros of
SB3's VecFrameStack. `depth=1` returns the plain features: the reactive baseline uses the same
chain as the driver with a memory.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from crashlearn.features import EGO_SIZE, LIDAR_RAYS
from crashlearn.history import DEPTH, CarHistory


class HistoryWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, depth: int = DEPTH):
        super().__init__(env)
        self.history = CarHistory(depth)
        size = depth * env.observation_space.shape[0]
        self.observation_space = spaces.Box(-1.0, 1.0, (size,), dtype=np.float32)

    def reset(self, **kwargs):
        self.history.reset()
        obs, info = self.env.reset(**kwargs)
        return self._stack(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._stack(obs, info), reward, terminated, truncated, info

    def _stack(self, obs: np.ndarray, info: dict) -> np.ndarray:
        return self.history.push(info["raw_obs"], info["step_info"], obs)


# friction_current in [0.7, 1.0] mapped to [-1, 1]
FRICTION_CENTER, FRICTION_HALF_RANGE = 0.85, 0.15


def friction_feature(info: dict) -> float:
    friction = info["step_info"]["friction_current"]
    return float(np.clip((friction - FRICTION_CENTER) / FRICTION_HALF_RANGE, -1.0, 1.0))


class FrictionWrapper(gym.Wrapper):
    """E-23 measurement only: appends friction_current, which the tournament info gives.

    Not part of the canonical features: a driver trained with it cannot be exported.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        size = env.observation_space.shape[0] + 1
        self.observation_space = spaces.Box(-1.0, 1.0, (size,), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._append(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._append(obs, info), reward, terminated, truncated, info

    def _append(self, obs: np.ndarray, info: dict) -> np.ndarray:
        return np.append(obs, np.float32(friction_feature(info)))


# sin and cos of the lap progress, then laps completed: after LiDAR, speed and steering.
# Kept out of features.py, whose source is copied into the submission agent.py.
PROGRESS_COLUMNS = slice(LIDAR_RAYS + 2, EGO_SIZE)
# Seeds the feature dropout apart from the episode streams of the env
DROPOUT_STREAM = 57


class InputMaskWrapper(gym.Wrapper):
    """E-57 training inputs, applied to each frame before `HistoryWrapper` stacks it.

    - `progress_blind`: the lap progress (sin, cos) and the laps completed are always zero.
      `train_ppo.py` also zeroes their first-layer weights at creation: with a zero input
      they receive no gradient and stay zero, so the exported model ignores the real values
      `agent.py` passes at the tournament. No correction after training (E-56).
    - `feature_dropout`: at every decision, each ego feature (LiDAR rays, speed, steering) is
      zeroed with this probability, so the driver cannot lean on a few rays. Training only.
    """

    def __init__(self, env: gym.Env, progress_blind: bool = False, feature_dropout: float = 0.0):
        super().__init__(env)
        if not 0.0 <= feature_dropout < 1.0:
            raise ValueError("feature_dropout is a probability, in [0, 1)")
        self.progress_blind = progress_blind
        self.feature_dropout = feature_dropout
        self._rng = np.random.default_rng()

    def reset(self, *, seed: int | None = None, **kwargs):
        if seed is not None:
            self._rng = np.random.default_rng([seed, DROPOUT_STREAM])
        obs, info = self.env.reset(seed=seed, **kwargs)
        return self._mask(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._mask(obs), reward, terminated, truncated, info

    def _mask(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.copy()
        if self.feature_dropout:
            drop = self._rng.random(EGO_SIZE) < self.feature_dropout
            obs[:EGO_SIZE][drop] = 0.0
        if self.progress_blind:
            obs[PROGRESS_COLUMNS] = 0.0
        return obs
