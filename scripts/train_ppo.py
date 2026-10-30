"""Train one PPO driver on the training tracks (E-18, docs/experience_e18.md).

`--history-depth 1` is the reactive baseline, `--history-depth 4` the driver with a memory:
every other setting is shared. The run directory keeps a manifest from the first second, so a
crashed run stays visible. An existing run is never overwritten: use a new `--output-dir`, or
`--resume` to continue it.
"""

import argparse
import json
import numbers
import sys
import time
import traceback
from pathlib import Path

from crashlearn.env import CrashLearnEnv
from crashlearn.evaluation import provenance, sha256
from crashlearn.history_env import HistoryWrapper

PPO_CONFIG = {
    "n_steps": 2048,  # per env
    "batch_size": 512,
    "n_epochs": 10,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    # log_std_init: the default std of 1 fills the whole [-1, 1] action range with noise
    "policy_kwargs": {"net_arch": [256, 256], "log_std_init": -1.0},
}


def make_env(depth: int):
    def build():
        return HistoryWrapper(CrashLearnEnv(), depth)

    return build


def attach_wandb(logger, run_name: str, config: dict) -> bool:
    """Mirror SB3's logger to W&B. The local CSV stays the reference if W&B is unreachable."""
    from stable_baselines3.common.logger import KVWriter

    class WandbWriter(KVWriter):
        def write(self, key_values, key_excluded, step=0):
            numeric = {k: v for k, v in key_values.items() if isinstance(v, numbers.Number)}
            wandb.log(numeric, step=step)

    try:
        import wandb

        wandb.init(project="crashlearn", name=run_name, config=config)
    except Exception as e:  # any W&B failure must leave the training running
        print(f"W&B disabled: {e}", file=sys.stderr)
        return False
    logger.output_formats.append(WandbWriter())
    return True


def train(args: argparse.Namespace, manifest: dict, manifest_path: Path) -> None:
    # The simulator ships an empty gym namespace, which SB3 reads when saving a model.
    import gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    if not hasattr(gym, "__version__"):
        gym.__version__ = "vendored-namespace"

    env = VecMonitor(SubprocVecEnv([make_env(args.history_depth)] * args.n_envs))
    try:
        if args.resume:
            model = PPO.load(args.resume, env=env, device=args.device)
        else:
            model = PPO("MlpPolicy", env, seed=args.seed, device=args.device, **PPO_CONFIG)
        logger = configure(str(args.output_dir), ["stdout", "csv"])
        if args.wandb:
            manifest["wandb"] = attach_wandb(logger, args.output_dir.name, manifest["config"])
        model.set_logger(logger)
        checkpoints = CheckpointCallback(
            save_freq=max(args.checkpoint_every // args.n_envs, 1),
            save_path=str(args.output_dir / "checkpoints"),
            name_prefix="ppo",
        )
        started = time.perf_counter()
        first_step = model.num_timesteps
        model.learn(
            total_timesteps=args.timesteps - first_step,
            callback=checkpoints,
            reset_num_timesteps=False,
        )
        elapsed = time.perf_counter() - started
        model.save(args.output_dir / "model.zip")
    finally:
        env.close()

    manifest |= {
        "status": "completed",
        "timesteps_done": model.num_timesteps,
        "wall_time_s": round(elapsed, 1),
        "steps_per_s": round((model.num_timesteps - first_step) / elapsed, 1),
        "model_sha256": sha256(args.output_dir / "model.zip"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-depth", type=int, required=True, choices=(1, 4))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-every", type=int, default=500_000)
    parser.add_argument("--resume", type=Path, help="checkpoint to continue from")
    parser.add_argument("--wandb", action="store_true", help="also log to W&B")
    args = parser.parse_args()
    if min(args.timesteps, args.n_envs, args.checkpoint_every) < 1:
        parser.error("--timesteps, --n-envs and --checkpoint-every must be positive")
    if args.resume and not args.resume.is_file():
        parser.error(f"no checkpoint at {args.resume}")

    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and not args.resume:
        parser.error(f"{args.output_dir} already holds a run: failures are kept, not replaced")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        **provenance(),
        "driver": f"ppo-h{args.history_depth}",
        "train_seed": args.seed,
        "history_depth": args.history_depth,
        "budget_timesteps": args.timesteps,
        "resumed_from": str(args.resume) if args.resume else None,
        "config": {**PPO_CONFIG, "n_envs": args.n_envs, "device": args.device},
        "status": "running",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        train(args, manifest, manifest_path)
    except Exception:
        manifest |= {"status": "failed", "error": traceback.format_exc()}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise
    print(f"{manifest['driver']} seed {args.seed}: {manifest['steps_per_s']} steps/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
