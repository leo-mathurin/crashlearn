"""Same races for every driver, failures included (docs/experiments.md).

No torch here: the CI runs this module, and a driver only needs `reset()` and
`act(obs, info) -> action`. Everything goes through `CrashLearnEnv`, the only path with
friction and LiDAR noise overrides, so a seed means the same race for every driver.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from crashlearn import tracks
from crashlearn.env import STATUS_DNF, STATUS_FINISHED, controls_to_action, sim
from crashlearn.scripted_driver import ScriptedDriver
from crashlearn.weather import WeatherConfig, schedule

DECISION_DT = 0.05  # s, 20 Hz
EVAL_SEEDS = (0, 1, 2)
# Subject v1.3.0 bounds: friction 1 = sunny, 0.7 = heavy rain; LiDAR noise half-range
# 0.001 = dry, 0.1 = heavy rain. `changing_weather` changes both during the race, as the
# tournament does, from a seeded schedule that training never draws (E-23, docs/weather.md).
# `strict_margin` applies the 0.50 m off-track tolerance of INSTRUCTIONS.md instead of the
# 1.00 m of the code, so that a driver does not depend on a margin the organizers may still
# fix (memlog, 2026-10-06).
STRICT_OFFTRACK_MARGIN_M = 0.5
# Seeds evaluation schedules from the race seed, apart from the training stream of the env.
EVAL_WEATHER_STREAM = 2301
CONDITIONS = {
    "nominal": {"friction": None, "lidar_noise": None},
    "low_friction": {"friction": 0.7, "lidar_noise": None},
    "lidar_noise": {"friction": None, "lidar_noise": 0.1},
    "heavy_rain": {"friction": 0.7, "lidar_noise": 0.1},
    "changing_weather": {"weather": WeatherConfig()},
    "strict_margin": {
        "friction": None,
        "lidar_noise": None,
        "offtrack_margin_m": STRICT_OFFTRACK_MARGIN_M,
    },
}
IDLE_BELOW_MPS = 0.1
REVERSE_BELOW_MPS = -0.1  # an idle car hovers around -1e-3 m/s
# credited progress beyond the distance driven in one decision: a cut through a wall
JUMP_ABOVE_M = 1.0


class Pilot(Protocol):
    def reset(self) -> None: ...

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray: ...


class ScriptedPilot:
    """The E-11 control driver, reading the raw observation a submission would receive."""

    def __init__(self) -> None:
        self.driver = ScriptedDriver()

    def reset(self) -> None:
        self.driver.reset()

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        return controls_to_action(*self.driver.predict(info["raw_obs"], info["step_info"]))


@dataclass(frozen=True)
class Race:
    split: str
    map: str
    condition: str
    seed: int


def campaign(
    maps: list[str] | None = None,
    conditions: list[str] | None = None,
    seeds: tuple[int, ...] = EVAL_SEEDS,
) -> list[Race]:
    """Nominal races on train and validation, perturbed ones on validation. Never the test."""
    nominal = [("train", tracks.train_tracks()), ("validation", tracks.validation_tracks())]
    perturbed = [("validation", tracks.validation_tracks())]
    return [
        Race(split, name, condition, seed)
        for condition in CONDITIONS
        for split, names in (nominal if condition == "nominal" else perturbed)
        for name in names
        for seed in seeds
        if (maps is None or name in maps) and (conditions is None or condition in conditions)
    ]


def run_race(env, pilot: Pilot, race: Race) -> dict:
    """One race to its end. DNF and TIMEOUT are results, reported like a finish."""
    settings = dict(CONDITIONS[race.condition])
    margin = settings.pop("offtrack_margin_m", None)
    default_margin = sim._OFFTRACK_MARGIN_M
    if margin is not None:
        # read at every step by the off-track check: set from our side, vendor/ untouched
        sim._OFFTRACK_MARGIN_M = margin
    try:
        return _race(env, pilot, race, settings)
    finally:
        sim._OFFTRACK_MARGIN_M = default_margin


def _race(env, pilot: Pilot, race: Race, settings: dict) -> dict:
    pilot.reset()
    if isinstance(settings.get("weather"), WeatherConfig):
        rng = np.random.default_rng([race.seed, EVAL_WEATHER_STREAM])
        settings["weather"] = schedule(settings["weather"], rng, env.unwrapped.max_episode_steps)
    obs, info = env.reset(seed=race.seed, options={"map": race.map, **settings})

    wall = vehicle = decisions = idle = reverse = jumps = 0
    distance = progress = total_reward = action_change = credited_m = 0.0
    previous = None
    trajectory = hashlib.sha256()
    done = False
    while not done:
        action = np.asarray(pilot.act(obs, info), dtype=np.float64)
        velocity = info["raw_obs"]["velocity"]
        driven = abs(velocity) * DECISION_DT
        distance += driven
        idle += abs(velocity) < IDLE_BELOW_MPS
        reverse += velocity < REVERSE_BELOW_MPS
        if previous is not None:
            action_change += float(np.sum((action - previous) ** 2))
        previous = action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        decisions += 1
        wall += info["wall_contact"]
        vehicle += info["vehicle_contact"]
        progress += info["progress_delta"]
        step_m = info["progress_delta"] * info["track_length_m"]
        credited_m += step_m
        jumps += step_m - driven > JUMP_ABOVE_M
        total_reward += reward
        trajectory.update(np.array([*action, info["progress_delta"]]).tobytes())

    status = {STATUS_FINISHED: "FINISHED", STATUS_DNF: "DNF"}.get(info["status"], "TIMEOUT")
    lap_times = list(info["step_info"]["lap_times"][env.unwrapped.ego_id])
    weather = settings.get("weather")
    return {
        **asdict(race),
        # the last values in effect; changing weather lists its segments below
        "friction": info["friction"],
        "lidar_noise": info["lidar_noise"],
        "weather_segments": [asdict(s) for s in weather] if weather else None,
        "status": status,
        "laps_completed": info["lap_count"],
        "progress_laps": round(progress, 4),
        "lap_times_s": lap_times,
        "total_time_s": round(sum(lap_times), 2) if status == "FINISHED" else None,
        "decisions": decisions,
        "wall_collision_steps": int(wall),
        "vehicle_collision_steps": int(vehicle),
        "distance_m": round(distance, 2),
        # credited minus driven: a racing line gains a little, a cut through a wall a lot
        "credited_minus_driven_m": round(credited_m - distance, 2),
        "progress_jump_steps": int(jumps),
        "idle_steps": int(idle),
        "reverse_steps": int(reverse),
        "action_change_per_step": round(action_change / decisions, 5),
        "offtrack_margin_m": float(sim._OFFTRACK_MARGIN_M),
        "dnf_step": info["step_count"] if status == "DNF" else None,
        "total_reward": round(total_reward, 2),
        "trajectory_sha256": trajectory.hexdigest(),
    }


def evaluate(env, pilot: Pilot, races: list[Race], output: Path, metadata: dict) -> list[dict]:
    """Rewrite `output` after every race: a crash mid-campaign keeps the finished races.

    Raises FileExistsError if `output` exists: earlier results are kept, not replaced.
    """
    if output.exists():
        raise FileExistsError(f"{output} already exists; preserve prior results")
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for race in races:
        records.append({**metadata, **run_race(env, pilot, race)})
        output.write_text(json.dumps(records, indent=2) + "\n")
    return records
