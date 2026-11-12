"""Race one driver through the evaluation campaign (docs/experiments.md).

Writes one JSON record per race, failures included. DNF and timeouts are results: exit code 0
whenever the races ran. The sealed test tracks are not part of the campaign.
"""

import argparse
import sys
from pathlib import Path

from crashlearn import MAX_DECISIONS
from crashlearn.env import OBS_SIZE, CrashLearnEnv
from crashlearn.evaluation import CONDITIONS, EVAL_SEEDS, ScriptedPilot, campaign, evaluate
from crashlearn.experiments import load_completed_run, provenance
from crashlearn.history_env import FrictionWrapper, HistoryWrapper


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
    manifest = load_completed_run(run_dir)
    model = run_dir / "model.zip"
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
        "derivation": manifest.get("derivation"),
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
    parser.add_argument(
        "--max-steps", type=int, default=MAX_DECISIONS, help="decisions before TIMEOUT"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve prior results")

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
    if (metadata.get("training_config") or {}).get("friction_feature"):
        env = FrictionWrapper(env)
    try:
        records = evaluate(env, pilot, races, args.output, metadata)
    finally:
        env.close()

    finished = sum(r["status"] == "FINISHED" for r in records)
    print(f"{metadata['driver']}: {finished}/{len(records)} finished, wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
