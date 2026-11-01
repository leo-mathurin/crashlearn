"""Same races for every driver, failures included (E-18, docs/experience_e18.md).

No torch here: the CI runs this module, and a driver only needs `reset()` and
`act(obs, info) -> action`. Everything goes through `CrashLearnEnv`, the only path with
friction and LiDAR noise overrides, so a seed means the same race for every driver.
"""

import datetime
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from crashlearn import SIM_DIR, tracks
from crashlearn.env import STATUS_DNF, STATUS_FINISHED, controls_to_action, sim
from crashlearn.reward import RewardConfig
from crashlearn.scripted_driver import ScriptedDriver

ROOT = Path(__file__).resolve().parents[2]
DECISION_DT = 0.05  # s, 20 Hz
EVAL_SEEDS = (0, 1, 2)
# Bounds documented in env_simulation.py: friction 1 = sunny, 0.5 = heavy rain; LiDAR noise
# half-range 0.001 = dry, 0.1 = heavy rain. `strict_margin` applies the 0.50 m off-track
# tolerance of INSTRUCTIONS.md instead of the 1.00 m of the code, so that a driver does not
# depend on a margin the organizers may still fix (memlog, 2026-10-06).
STRICT_OFFTRACK_MARGIN_M = 0.5
CONDITIONS = {
    "nominal": {"friction": None, "lidar_noise": None},
    "low_friction": {"friction": 0.5, "lidar_noise": None},
    "lidar_noise": {"friction": None, "lidar_noise": 0.1},
    "heavy_rain": {"friction": 0.5, "lidar_noise": 0.1},
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
BASELINE = "ppo-h1"
CANDIDATE = "ppo-h4"
BOOTSTRAP_SAMPLES = 10_000
# Below this, a bootstrap interval only describes the few seeds it resamples.
MIN_SEEDS = 3


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


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance() -> dict:
    status = _git("status", "--porcelain", "--untracked-files=no")
    package = Path(__file__).parent
    return {
        # the training image has no git: the station passes the commit through the environment
        "commit": os.environ.get("GIT_COMMIT") or _git("rev-parse", "HEAD"),
        "dirty": None if status == "unknown" else bool(status),
        "sim_sha256": sha256(SIM_DIR / "env_simulation.py"),
        "features_sha256": sha256(package / "features.py"),
        "history_sha256": sha256(package / "history.py"),
        "reward_sha256": sha256(package / "reward.py"),
        "reward_version": RewardConfig().version,
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }


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
    return {
        **asdict(race),
        "friction": info["friction"],
        "lidar_noise": info["lidar_noise"],
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
    """Rewrite `output` after every race: a crash mid-campaign keeps the finished races."""
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for race in races:
        records.append({**metadata, **run_race(env, pilot, race)})
        output.write_text(json.dumps(records, indent=2) + "\n")
    return records


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def bootstrap_interval(values: list[float], seed: int = 0) -> tuple[float, float] | None:
    """95 % bootstrap interval of the mean, None with fewer than MIN_SEEDS values."""
    if len(values) < MIN_SEEDS:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (BOOTSTRAP_SAMPLES, len(values))).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_differences(
    records: list[dict], baseline: str = BASELINE, candidate: str = CANDIDATE
) -> dict:
    """Candidate minus baseline per training seed, on the races both drivers ran.

    Time gains only use the races both drivers finished: comparing all finishes would reward
    a driver for failing on the hard tracks.
    """
    runs: dict[tuple, dict] = {}
    for r in records:
        key = (r["map"], r["condition"], r["seed"])
        runs.setdefault((r["driver"], r["train_seed"]), {})[key] = r

    seeds = sorted(
        s for driver, s in runs if driver == candidate and (baseline, s) in runs and s is not None
    )
    finish, gains = {}, {}
    for seed in seeds:
        base, cand = runs[baseline, seed], runs[candidate, seed]
        shared = sorted(base.keys() & cand.keys())
        if not shared:
            continue
        pairs = [(base[k], cand[k]) for k in shared]
        finish[seed] = _mean(
            [(c["status"] == "FINISHED") - (b["status"] == "FINISHED") for b, c in pairs]
        )
        both = [
            b["total_time_s"] - c["total_time_s"]
            for b, c in pairs
            if b["status"] == c["status"] == "FINISHED"
        ]
        if both:
            gains[seed] = _mean(both)
    return {"finish_rate": finish, "time_gain_s": gains}


def verdict(finish_interval: tuple | None, time_interval: tuple | None) -> str:
    """The rule fixed before the campaign: only an interval excluding 0 concludes."""
    if finish_interval is None:
        return "non concluant"
    if finish_interval[0] > 0:
        return "h4 plus fiable"
    if finish_interval[1] < 0:
        return "h4 moins fiable"
    if time_interval is None:
        return "non concluant"
    if time_interval[0] > 0:
        return "h4 plus rapide"
    if time_interval[1] < 0:
        return "h4 plus lent"
    return "non concluant"


def scopes(records: list[dict]) -> list[tuple[str, str]]:
    order = {name: i for i, name in enumerate(CONDITIONS)}
    return sorted(
        {(r["split"], r["condition"]) for r in records}, key=lambda s: (s[0], order[s[1]])
    )


def driver_summary(records: list[dict]) -> dict:
    """Per-race means, plus the finish rate averaged per map so that no map weighs more."""
    finished = [r for r in records if r["status"] == "FINISHED"]
    maps = sorted({r["map"] for r in records})
    per_map = [_mean([r["status"] == "FINISHED" for r in records if r["map"] == m]) for m in maps]
    distance = sum(r["distance_m"] for r in records)

    def per_100m(key: str) -> float | None:
        return 100 * sum(r[key] for r in records) / distance if distance else None

    return {
        "races": len(records),
        "finish_rate": len(finished) / len(records),
        "finish_rate_per_map": _mean(per_map),
        "dnf": sum(r["status"] == "DNF" for r in records),
        "timeout": sum(r["status"] == "TIMEOUT" for r in records),
        "median_time_s": float(np.median([r["total_time_s"] for r in finished]))
        if finished
        else None,
        "wall_steps_per_race": _mean([r["wall_collision_steps"] for r in records]),
        "wall_steps_per_100m": per_100m("wall_collision_steps"),
        "vehicle_steps_per_100m": per_100m("vehicle_collision_steps"),
        "idle_steps_per_race": _mean([r["idle_steps"] for r in records]),
        "reverse_steps_per_race": _mean([r["reverse_steps"] for r in records]),
        "action_change_per_step": _mean([r["action_change_per_step"] for r in records]),
        "progress_jumps_per_race": _mean([r["progress_jump_steps"] for r in records]),
        "credited_minus_driven_m": _mean([r["credited_minus_driven_m"] for r in records]),
    }
