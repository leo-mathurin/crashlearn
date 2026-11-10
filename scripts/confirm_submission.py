"""Confirm a frozen solo submission on validation maps in the tournament CPU image.

Loads the agent with the tournament loader and runs its controls through the native
simulator, without Gymnasium or a training framework. A race seed gives the same engine
seed as in CrashLearnEnv.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from crashlearn import MAX_DECISIONS, add_simulator_to_path, engine_seed, map_cache
from crashlearn.reward import STATUS_ACTIVE, STATUS_DNF, STATUS_FINISHED
from crashlearn.tracks import validation_tracks

add_simulator_to_path()
import env_simulation as sim  # noqa: E402
from agent_loader import validate_and_load_agent  # noqa: E402

STATUS = {STATUS_DNF: "DNF", STATUS_ACTIVE: "TIMEOUT", STATUS_FINISHED: "FINISHED"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-count", type=int, default=30)
    args = parser.parse_args()
    if args.seed_count < 1:
        parser.error("seed-count must be positive")
    if args.output.exists():
        parser.error("output already exists; preserve prior results")
    # The loader checks the submission like the tournament; each race gets a fresh Agent.
    agent_class = type(validate_and_load_agent(args.submission)[1])
    exported = json.loads((args.submission / "export.json").read_text())
    map_cache.install()
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for track in validation_tracks():
            sim.set_map(track)
            for seed in range(args.seed_start, args.seed_start + args.seed_count):
                race_engine_seed = engine_seed(np.random.default_rng(seed))
                sim.reset(1, seed=race_engine_seed)
                sim._get_sim()._last_scan_noise = None
                agent = agent_class()
                wall = 0
                progress = 0.0
                maximum_speed = float("-inf")
                for _ in range(MAX_DECISIONS):
                    obs, info = sim.get_obs(0), sim.get_step_info()
                    speed, steer = agent.predict(obs, info)
                    maximum_speed = max(maximum_speed, speed)
                    sim.apply_action(0, speed, steer)
                    sim.simulation_step()
                    info = sim.get_step_info()
                    wall += bool(info["collisions"]["wall"][0])
                    progress += info["progress_delta"][0]
                    if info["agent_status"][0] != STATUS_ACTIVE:
                        break
                status = STATUS[info["agent_status"][0]]
                records.append(
                    {
                        "map": track,
                        "split": "validation",
                        "condition": "nominal",
                        "seed": seed,
                        "engine_seed": race_engine_seed,
                        "status": status,
                        "progress_laps": round(progress, 4),
                        "decisions": info["step_count"],
                        "lap_times_s": list(info["lap_times"][0]),
                        "wall_collision_steps": wall,
                        "max_target_speed_mps": maximum_speed,
                        "source_model_sha256": exported["source_model_sha256"],
                        "torch_imported": "torch" in sys.modules,
                        "stable_baselines3_imported": "stable_baselines3" in sys.modules,
                    }
                )
                args.output.write_text(json.dumps(records, indent=2) + "\n")
            print(
                f"{track}: {sum(r['status'] == 'FINISHED' for r in records if r['map'] == track)}"
                f"/{args.seed_count} finished",
                flush=True,
            )
    finally:
        sim.close()
    assert "torch" not in sys.modules and "stable_baselines3" not in sys.modules
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
