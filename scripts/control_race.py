"""Control races of the scripted diagnostic driver on training tracks.

Writes one JSON record per car (with provenance) and prints a summary. DNF and timeouts are
results too: exit code 0 whenever the races ran, 2 on invalid arguments.
Validation and sealed test tracks are refused (docs/tracks.md).
"""

import argparse
import dataclasses
import datetime
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from crashlearn import SIM_DIR, add_simulator_to_path, tracks
from crashlearn.scripted_driver import DriverConfig, ScriptedDriver

DEFAULT_MAPS = ["IMS", "BrandsHatch", "Austin", "MexicoCity", "Budapest"]
DECISION_DT = 0.05  # s, 20 Hz
SIM_REQUIRED_LAPS = 3  # env_simulation freezes a car as FINISHED at 3 laps
ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def duration(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < DECISION_DT:
        raise argparse.ArgumentTypeError(f"must be finite and >= {DECISION_DT} s, got {text!r}")
    return value


def provenance(args: argparse.Namespace, config: DriverConfig) -> dict:
    sim_file = SIM_DIR / "env_simulation.py"
    setup = (SIM_DIR / "setup.py").read_text()
    status = git("status", "--porcelain", "--untracked-files=no")  # untracked files ignored
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": None if status == "unknown" else bool(status),
        "sim_sha256": hashlib.sha256(sim_file.read_bytes()).hexdigest(),
        "f110_gym_version": re.search(r"version=['\"]([^'\"]+)", setup).group(1),
        "seed": args.seed,
        "cars": args.cars,
        "laps": args.laps,
        "max_time_s": args.max_time,
        "driver_config": dataclasses.asdict(config),
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }


def run_race(sim, name: str, args: argparse.Namespace, config: DriverConfig) -> list[dict]:
    sim.close()  # fresh singleton: its noise RNG is only created at construction
    sim.set_map(name)
    sim.reset(num_cars=args.cars)
    # ponytail: private hook, the simulator exposes no seed (E-12); only drives LiDAR noise
    sim._get_sim()._rng = np.random.default_rng(args.seed)

    drivers = [ScriptedDriver(config) for _ in range(args.cars)]
    cars = [
        {"wall": 0, "vehicle": 0, "decisions": 0, "distance": 0.0, "end_step": None}
        for _ in range(args.cars)
    ]
    trajectory = hashlib.sha256()
    frictions = set()
    info = sim.get_step_info()
    for _ in range(round(args.max_time / DECISION_DT)):
        running = [
            i
            for i, car in enumerate(cars)
            if car["end_step"] is None and info["agent_status"][i] == 1
        ]
        if not running:
            break
        for i in range(args.cars):
            if i not in running:
                sim.apply_action(i, 0.0, 0.0)
                continue
            obs = sim.get_obs(i)
            cars[i]["distance"] += abs(obs["velocity"]) * DECISION_DT
            speed, steer = drivers[i].predict(obs, info)
            trajectory.update(np.array([i, speed, steer, obs["progress"]]).tobytes())
            sim.apply_action(i, speed, steer)
        sim.simulation_step()
        info = sim.get_step_info()
        frictions.add(info["friction_current"])
        for i in running:
            car = cars[i]
            car["decisions"] += 1
            car["wall"] += info["collisions"]["wall"][i]
            car["vehicle"] += info["collisions"]["vehicle"][i]
            if sim.get_obs(i)["lap_count"] >= args.laps or info["agent_status"][i] != 1:
                car["end_step"] = info["step_count"]

    records = []
    for i, car in enumerate(cars):
        obs = sim.get_obs(i)
        laps_done = min(obs["lap_count"], args.laps)
        if laps_done >= args.laps:
            status = "FINISHED"
        elif info["agent_status"][i] == 0:
            status = "DNF"
        else:
            status = "TIMEOUT"
        lap_times = info["lap_times"][i][: args.laps]
        records.append(
            {
                "map": name,
                "car": i,
                "status": status,
                "laps_completed": laps_done,
                "progress_laps": round(obs["lap_count"] + obs["progress"], 4),
                "lap_times_s": lap_times,
                "total_time_s": round(sum(lap_times), 2) if status == "FINISHED" else None,
                "decisions": car["decisions"],
                "wall_collision_steps": car["wall"],
                "vehicle_collision_steps": car["vehicle"],
                "distance_m": round(car["distance"], 2),
                "dnf_step": car["end_step"] if status == "DNF" else None,
                "friction_seen": sorted(frictions),
                "race_trajectory_sha256": trajectory.hexdigest(),
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map",
        action="append",
        dest="maps",
        help=f"training track, repeatable (default: {' '.join(DEFAULT_MAPS)})",
    )
    parser.add_argument("--seed", type=int, default=0, help="LiDAR noise seed (default 0)")
    parser.add_argument(
        "--laps",
        type=int,
        default=SIM_REQUIRED_LAPS,
        choices=range(1, SIM_REQUIRED_LAPS + 1),
        help="laps per race; internal evaluations use 3 (default 3)",
    )
    parser.add_argument("--cars", type=int, default=1, choices=range(1, 5), help="1 to 4 cars")
    parser.add_argument(
        "--max-time", type=duration, default=600.0, help="simulated seconds before TIMEOUT"
    )
    parser.add_argument("--output", type=Path, default=Path("runs/control_race.json"))
    args = parser.parse_args()
    args.maps = args.maps or DEFAULT_MAPS

    train = tracks.train_tracks()
    for name in args.maps:
        if name not in train:
            parser.error(f"{name!r} is not a canonical training track: {', '.join(train)}")
    try:  # fail now, not after an hour of racing; "a" keeps an existing file intact
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.open("a").close()
    except OSError as e:
        parser.error(f"cannot write --output {args.output}: {e}")

    add_simulator_to_path()
    import env_simulation as sim

    config = DriverConfig()
    prov = provenance(args, config)
    records = []
    try:
        for name in args.maps:
            records += [{**prov, **r} for r in run_race(sim, name, args, config)]
            # rewritten after every race: a crash mid-campaign keeps the finished races
            args.output.write_text(json.dumps(records, indent=2) + "\n")
    finally:
        sim.close()

    print(f"commit {prov['commit'][:12]}{' (dirty)' if prov['dirty'] else ''}  seed {args.seed}")
    print(f"driver v{config.version}  sim {prov['sim_sha256'][:12]}  laps {args.laps}\n")
    print(
        f"{'map':<12} {'car':>3} {'status':<8} {'laps':>4} {'progress':>8} {'total_s':>8} "
        f"{'wall':>5} {'veh':>4} {'dist_m':>8}  lap_times_s"
    )
    for r in records:
        total = f"{r['total_time_s']:.2f}" if r["total_time_s"] is not None else "-"
        print(
            f"{r['map']:<12} {r['car']:>3} {r['status']:<8} {r['laps_completed']:>4} "
            f"{r['progress_laps']:>8.4f} {total:>8} {r['wall_collision_steps']:>5} "
            f"{r['vehicle_collision_steps']:>4} {r['distance_m']:>8.1f}  {r['lap_times_s']}"
        )
    print(f"\nwrote {len(records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
