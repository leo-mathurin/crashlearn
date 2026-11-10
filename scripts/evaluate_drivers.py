"""Race one driver through the E-18 campaign (docs/experience_e18.md).

Writes one JSON record per race, failures included. DNF and timeouts are results: exit code 0
whenever the races ran. The sealed test tracks are not part of the campaign.
"""

import argparse
import json
import sys
from pathlib import Path

from crashlearn.env import OBS_SIZE, CrashLearnEnv
from crashlearn.evaluation import (
    CONDITIONS,
    EVAL_SEEDS,
    ScriptedPilot,
    campaign,
    evaluate,
    provenance,
    sha256,
)
from crashlearn.history_env import HistoryWrapper


class PPOPilot:
    def __init__(self, path: Path):
        from stable_baselines3 import PPO

        self.model = PPO.load(path, device="cpu")
        self.depth = self.model.observation_space.shape[0] // OBS_SIZE

    def reset(self) -> None:
        pass

    def act(self, obs, info):
        return self.model.predict(obs, deterministic=True)[0]


def load_run(run_dir: Path) -> tuple[PPOPilot, dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError(f"{run_dir}: run status is {manifest['status']!r}")
    current = provenance()
    for key in ("features_sha256", "history_sha256"):
        if manifest[key] != current[key]:
            raise ValueError(f"{run_dir}: training has a different {key}; use its original sources")
    model = run_dir / "model.zip"
    if sha256(model) != manifest["model_sha256"]:
        raise ValueError(f"{model} does not match its manifest")
    pilot = PPOPilot(model)
    if pilot.depth != manifest["history_depth"]:
        raise ValueError(f"{model}: depth {pilot.depth}, manifest {manifest['history_depth']}")
    metadata = {
        "driver": manifest["driver"],
        "train_seed": manifest["train_seed"],
        "history_depth": pilot.depth,
        "model_sha256": manifest["model_sha256"],
        "budget_timesteps": manifest["budget_timesteps"],
        "training_config": manifest["config"],
        "initialization": manifest.get("initialization"),
    }
    return pilot, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="directory written by train_ppo.py")
    source.add_argument("--scripted", action="store_true", help="the E-11 control driver")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map", action="append", dest="maps", help="restrict the campaign")
    parser.add_argument("--condition", action="append", choices=CONDITIONS, dest="conditions")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--max-steps", type=int, default=8000, help="decisions before TIMEOUT")
    args = parser.parse_args()

    races = campaign(args.maps, args.conditions, tuple(args.seeds or EVAL_SEEDS))
    if not races:
        parser.error("no race left: --map must name a training or validation track")

    if args.scripted:
        pilot = ScriptedPilot()
        metadata = {"driver": "scripted", "train_seed": None, "history_depth": 1}
    else:
        pilot, metadata = load_run(args.run_dir)
    metadata |= {**provenance(), "max_steps": args.max_steps}

    env = HistoryWrapper(CrashLearnEnv(max_episode_steps=args.max_steps), metadata["history_depth"])
    try:
        records = evaluate(env, pilot, races, args.output, metadata)
    finally:
        env.close()

    finished = sum(r["status"] == "FINISHED" for r in records)
    print(f"{metadata['driver']}: {finished}/{len(records)} finished, wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
