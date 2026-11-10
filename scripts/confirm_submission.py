"""Confirm a frozen solo submission on validation maps in the tournament CPU image.

Runs the actual Agent controls through the native simulator, without Gymnasium or a
training framework. The race seed maps to the same engine seed as CrashLearnEnv.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from crashlearn import add_simulator_to_path
from crashlearn.map_cache import cached_map_loader
from crashlearn.tracks import validation_tracks

add_simulator_to_path()
import env_simulation as sim  # noqa: E402
from f110_gym.envs.laser_models import ScanSimulator2D  # noqa: E402


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
    spec = importlib.util.spec_from_file_location("submission_agent", args.submission / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    exported = json.loads((args.submission / "export.json").read_text())
    ScanSimulator2D.set_map = cached_map_loader(ScanSimulator2D.set_map)
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for track in validation_tracks():
            sim.set_map(track)
            for seed in range(args.seed_start, args.seed_start + args.seed_count):
                engine_seed = int(np.random.default_rng(seed).integers(2**31))
                sim.reset(1, seed=engine_seed)
                sim._get_sim()._last_scan_noise = None
                agent = module.Agent()
                wall = 0
                progress = 0.0
                maximum_speed = float("-inf")
                for _ in range(8000):
                    obs, info = sim.get_obs(0), sim.get_step_info()
                    speed, steer = agent.predict(obs, info)
                    maximum_speed = max(maximum_speed, speed)
                    sim.apply_action(0, speed, steer)
                    sim.simulation_step()
                    info = sim.get_step_info()
                    wall += bool(info["collisions"]["wall"][0])
                    progress += info["progress_delta"][0]
                    if info["agent_status"][0] != 1:
                        break
                status = {0: "DNF", 1: "TIMEOUT", 2: "FINISHED"}[info["agent_status"][0]]
                records.append(
                    {
                        "map": track,
                        "split": "validation",
                        "condition": "nominal",
                        "seed": seed,
                        "engine_seed": engine_seed,
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
