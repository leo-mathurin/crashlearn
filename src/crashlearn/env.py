"""Gymnasium wrapper around the vendored simulator (E-14).

One learning car (`ego_id`) per env; the other cars, if any, are driven by `opponent_policy`.
The simulator is a process-wide singleton: a single env may be open per process, so
vectorize with `SubprocVecEnv`, never `DummyVecEnv`.

Episode ends, decided here since the simulator has no `done`:
- terminated: the ego car is FINISHED (3 laps) or DNF (stagnation: < 1 m in 10 s, or off the
  tolerated track corridor). A wall or vehicle contact alone never ends the episode.
- truncated: `max_episode_steps` decisions (20 Hz) reached first.

Observation (E-15 and E-16 will refine it): 105 floats in [-1, 1] = 100 LiDAR rays / 15 m,
velocity / 10 m/s, steering / 0.4189 rad, sin and cos of the lap phase, laps / 3.
Action: two floats in [-1, 1], mapped affinely to target speed [-2, 10] m/s and steering
[-0.4189, 0.4189] rad.

Friction and LiDAR noise: the official simulator pins them (friction 1.0, noise ±0.1 %, `TODO`
upstream). `friction` and `lidar_noise` override them from our side, `vendor/` untouched;
`None` keeps the official behaviour, i.e. the tournament conditions.
"""

import math
from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from crashlearn import add_simulator_to_path
from crashlearn.tracks import train_tracks

add_simulator_to_path()
import env_simulation as sim  # noqa: E402

SPEED_MIN, SPEED_MAX = -2.0, 10.0
STEER_MAX = sim._PARAMS["s_max"]
LIDAR_MAX = sim._LIDAR_MAX
LIDAR_RAYS = sim._LIDAR_RAYS
REQUIRED_LAPS = sim._REQUIRED_LAPS
OBS_SIZE = LIDAR_RAYS + 5

STATUS_DNF, STATUS_ACTIVE, STATUS_FINISHED = 0, 1, 2

# (raw simulator obs, raw step info, ego_id) -> reward. Default: progress only, E-15 calibrates.
RewardFn = Callable[[dict, dict, int], float]
# raw simulator obs of one opponent car -> (target_speed m/s, steering rad)
OpponentPolicy = Callable[[dict], tuple[float, float]]
# fixed value, or (low, high) sampled uniformly at each reset
Setting = float | tuple[float, float] | None

_open_env: "CrashLearnEnv | None" = None


def progress_reward(obs: dict, info: dict, ego_id: int) -> float:
    """1000 x signed lap fraction gained this step (see the MDP document)."""
    return 1000.0 * info["progress_delta"][ego_id]


def action_to_controls(action: np.ndarray) -> tuple[float, float]:
    """Map a normalized action in [-1, 1]^2 to (target_speed m/s, steering rad)."""
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    speed = SPEED_MIN + (a[0] + 1.0) * 0.5 * (SPEED_MAX - SPEED_MIN)
    return float(speed), float(a[1] * STEER_MAX)


def controls_to_action(speed: float, steering: float) -> np.ndarray:
    """Inverse of action_to_controls, for scripted drivers run through the env."""
    a0 = 2.0 * (speed - SPEED_MIN) / (SPEED_MAX - SPEED_MIN) - 1.0
    return np.clip([a0, steering / STEER_MAX], -1.0, 1.0).astype(np.float32)


def encode_observation(obs: dict) -> np.ndarray:
    """Raw simulator obs of one car -> float32 vector of OBS_SIZE values in [-1, 1]."""
    phase = 2.0 * math.pi * obs["progress"]
    tail = [
        obs["velocity"] / SPEED_MAX,
        obs["steering"] / STEER_MAX,
        math.sin(phase),
        math.cos(phase),
        obs["lap_count"] / REQUIRED_LAPS,
    ]
    vec = np.concatenate([np.asarray(obs["lidar"], dtype=np.float64) / LIDAR_MAX, tail])
    return np.clip(vec, -1.0, 1.0).astype(np.float32)


def _sample(setting: Setting, rng: np.random.Generator) -> float | None:
    if setting is None or isinstance(setting, int | float):
        return setting
    low, high = setting
    return float(rng.uniform(low, high))


class CrashLearnEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        num_cars: int = 1,
        maps: str | Sequence[str] | None = None,
        ego_id: int = 0,
        max_episode_steps: int = 8000,
        reward_fn: RewardFn = progress_reward,
        opponent_policy: OpponentPolicy | None = None,
        friction: Setting = None,
        lidar_noise: Setting = None,
    ):
        global _open_env
        if _open_env is not None:
            raise RuntimeError("the simulator is a process-wide singleton: close the open env")
        if not 1 <= num_cars <= sim._MAX_SLOTS:
            raise ValueError(f"num_cars must be in [1, {sim._MAX_SLOTS}], got {num_cars}")
        if not 0 <= ego_id < num_cars:
            raise ValueError(f"ego_id must be in [0, {num_cars}), got {ego_id}")

        self.num_cars = num_cars
        self.maps = (maps,) if isinstance(maps, str) else tuple(maps or train_tracks())
        unknown = set(self.maps) - set(sim.get_available_maps()) - {"example"}
        if unknown:
            raise ValueError(f"unknown maps: {sorted(unknown)}")
        self.ego_id = ego_id
        self.max_episode_steps = max_episode_steps
        self.reward_fn = reward_fn
        self.opponent_policy = opponent_policy
        self.friction = friction
        self.lidar_noise = lidar_noise

        self.observation_space = spaces.Box(-1.0, 1.0, (OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

        self._noise_rng = np.random.default_rng()
        self._episode_noise: float | None = None
        self._episode_friction: float | None = None
        self._map: str | None = None
        _open_env = self

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        options = options or {}

        name = options.get("map") or self.maps[int(self.np_random.integers(len(self.maps)))]
        sim.set_map(name)
        self._map = name

        # One seed per episode from np_random: seeding once makes every later episode
        # replayable. It rebuilds the engine (a few ms) and re-arms its noise stream.
        episode_seed = int(self.np_random.integers(2**31))
        sim.reset(self.num_cars, seed=episode_seed)
        self._noise_rng = np.random.default_rng(episode_seed)

        self._episode_friction = _sample(options.get("friction", self.friction), self.np_random)
        self._episode_noise = _sample(options.get("lidar_noise", self.lidar_noise), self.np_random)
        if self._episode_friction is not None:
            params = dict(sim._PARAMS, mu=self._episode_friction)
            # upstream _update_friction never re-applies mu: this holds for the episode
            sim._get_sim()._sim.update_params(params, agent_idx=-1)
        # The simulator keeps the last step's noise across reset: without this, the first
        # obs of an episode would depend on the previous one. None = noiseless, as in a
        # fresh process.
        sim._get_sim()._last_scan_noise = self._draw_noise()

        raw = sim.get_obs(self.ego_id)
        return encode_observation(raw), self._info(raw, sim.get_step_info())

    def step(self, action):
        speed, steering = action_to_controls(action)
        sim.apply_action(self.ego_id, speed, steering)
        if self.opponent_policy is not None:
            for car in range(self.num_cars):
                if car != self.ego_id:
                    sim.apply_action(car, *self.opponent_policy(sim.get_obs(car)))
        sim.simulation_step()
        if self._episode_noise is not None:
            sim._get_sim()._last_scan_noise = self._draw_noise()

        raw = sim.get_obs(self.ego_id)
        step_info = sim.get_step_info()
        reward = float(self.reward_fn(raw, step_info, self.ego_id))
        terminated = step_info["agent_status"][self.ego_id] != STATUS_ACTIVE
        truncated = not terminated and step_info["step_count"] >= self.max_episode_steps
        return encode_observation(raw), reward, terminated, truncated, self._info(raw, step_info)

    def close(self):
        global _open_env
        if _open_env is self:
            sim.close()
            _open_env = None

    def _draw_noise(self) -> np.ndarray | None:
        if self._episode_noise is None:
            return None
        h = self._episode_noise
        return self._noise_rng.uniform(-h, h, size=LIDAR_RAYS)

    def _info(self, raw: dict, step_info: dict) -> dict:
        i = self.ego_id
        return {
            "map": self._map,
            "step_count": step_info["step_count"],
            "status": step_info["agent_status"][i],
            "lap_count": raw["lap_count"],
            "lap_complete": step_info["lap_complete"][i],
            "progress_delta": step_info["progress_delta"][i],
            "wall_contact": step_info["collisions"]["wall"][i],
            "vehicle_contact": step_info["collisions"]["vehicle"][i],
            "rank": step_info["ranks"][i],
            "friction": self._episode_friction,
            "lidar_noise": self._episode_noise,
            "raw_obs": raw,
            "step_info": step_info,
        }
