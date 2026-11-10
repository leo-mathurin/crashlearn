"""Train a PPO driver with a persistent manifest, checkpoints, CSV and optional W&B.

Depth 1 is the reactive baseline; depth 4 has memory. Resume into the original output
folder with a larger target budget. Each attempt keeps its own logs. Environment RNG and
partial rollouts are not restored, so a resumed run is not an exact replay.
"""

import argparse
import signal
import sys
import time
import traceback
from pathlib import Path

from crashlearn.env import CrashLearnEnv
from crashlearn.evaluation import sha256
from crashlearn.experiments import attach_wandb, now, prepare_run, write_manifest
from crashlearn.history_env import HistoryWrapper
from crashlearn.map_cache import cached_map_loader
from crashlearn.tracks import train_tracks

PPO_CONFIG = {
    "n_steps": 2048,
    "batch_size": 512,
    "n_epochs": 10,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "policy_kwargs": {"net_arch": [256, 256], "log_std_init": -1.0},
}


def make_env(
    depth: int, track: str | None = None, max_speed: float = 10.0, random_start: bool = False
):
    def build():
        from f110_gym.envs.laser_models import ScanSimulator2D

        ScanSimulator2D.set_map = cached_map_loader(ScanSimulator2D.set_map)
        return HistoryWrapper(
            CrashLearnEnv(maps=track, max_speed=max_speed, random_start=random_start), depth
        )

    return build


def train(args: argparse.Namespace, manifest: dict, manifest_path: Path) -> None:
    import gym
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    # The simulator ships an empty gym namespace, which SB3 reads when saving a model.
    if not hasattr(gym, "__version__"):
        gym.__version__ = "vendored-namespace"
    torch.set_num_threads(args.torch_threads)
    maps = train_tracks()
    builds = [
        make_env(
            args.history_depth,
            maps[i % len(maps)] if args.fixed_map_workers else None,
            args.max_speed,
            args.random_start,
        )
        for i in range(args.n_envs)
    ]
    env = VecMonitor(SubprocVecEnv(builds))
    model = logger = writer = None
    attempt = manifest["attempts"][-1]
    started = time.perf_counter()
    try:
        if args.resume:
            model = PPO.load(args.resume, env=env, device=args.device)
            attempt["checkpoint_sha256"] = sha256(args.resume)
            if args.timesteps <= model.num_timesteps:
                raise ValueError("target --timesteps must exceed the checkpoint step")
            if model.seed != args.seed:
                raise ValueError("checkpoint seed differs from the run seed")
        else:
            model = PPO("MlpPolicy", env, seed=args.seed, device=args.device, **PPO_CONFIG)
            if args.initialize_from:
                source = PPO.load(args.initialize_from, device=args.device)
                if source.observation_space.shape != model.observation_space.shape:
                    raise ValueError("initialization checkpoint has a different observation shape")
                model.policy.load_state_dict(source.policy.state_dict())
                manifest["initialization"] = {
                    "checkpoint": str(args.initialize_from),
                    "checkpoint_sha256": sha256(args.initialize_from),
                    "source_timesteps": source.num_timesteps,
                    "optimizer_restored": False,
                }
                model.set_random_seed(args.seed)
                del source
        args.log_dir = args.output_dir / "logs" / f"attempt-{len(manifest['attempts']):03d}"
        logger = configure(str(args.log_dir), ["stdout", "csv"])
        if args.wandb:
            writer = attach_wandb(logger, args, manifest)
        model.set_logger(logger)
        attempt["first_step"] = first_step = model.num_timesteps
        attempt["log_dir"] = str(args.log_dir.relative_to(args.output_dir))
        attempt["device"] = str(model.device)
        attempt["cuda_version"] = torch.version.cuda
        attempt["gpu"] = (
            torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else None
        )
        write_manifest(manifest_path, manifest)

        class TrackedCheckpoint(CheckpointCallback):
            def _on_step(self):
                result = super()._on_step()
                if writer is not None and self.n_calls % self.save_freq == 0:
                    writer.checkpoint(Path(self._checkpoint_path(extension="zip")))
                return result

        checkpoints = TrackedCheckpoint(
            save_freq=max(args.checkpoint_every // args.n_envs, 1),
            save_path=str(args.output_dir / "checkpoints"),
            name_prefix=f"ppo_attempt{len(manifest['attempts']):03d}",
        )
        model.learn(
            total_timesteps=args.timesteps - first_step,
            callback=checkpoints,
            reset_num_timesteps=False,
        )
        model.save(args.output_dir / "model.zip")
        if writer is not None:
            writer.checkpoint(args.output_dir / "model.zip")
        manifest.pop("error", None)
        manifest["model_sha256"] = sha256(args.output_dir / "model.zip")
        elapsed = time.perf_counter() - started
        manifest |= {
            "status": "completed",
            "timesteps_done": model.num_timesteps,
            "wall_time_s": round(elapsed, 2),
            "steps_per_s": round((model.num_timesteps - first_step) / elapsed, 1),
        }
        attempt |= {
            "status": "completed",
            "last_step": model.num_timesteps,
            "wall_time_s": manifest["wall_time_s"],
        }
    except (Exception, KeyboardInterrupt) as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        attempt |= {
            "status": status,
            "error": traceback.format_exc(),
            "wall_time_s": round(time.perf_counter() - started, 2),
        }
        manifest |= {"status": status, "error": attempt["error"]}
        if model is not None:
            attempt["last_step"] = model.num_timesteps
        raise
    finally:
        try:
            try:
                if logger is not None:
                    logger.close()
            finally:
                env.close()
        except Exception:
            attempt["cleanup_error"] = traceback.format_exc()
            manifest["status"] = attempt["status"] = "failed"
            manifest.setdefault("error", attempt["cleanup_error"])
            raise
        finally:
            attempt["finished_utc"] = now()
            manifest["total_wall_time_s"] = round(
                sum(a.get("wall_time_s", 0) for a in manifest["attempts"]), 2
            )
            write_manifest(manifest_path, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-depth", type=int, required=True, choices=(1, 4))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--fixed-map-workers",
        action="store_true",
        help="one training track per worker; n-envs must be a multiple of 15",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume", type=Path, help="checkpoint in the original run")
    source.add_argument(
        "--initialize-from",
        type=Path,
        help="start a new run from compatible policy weights, with a fresh optimizer",
    )
    parser.add_argument("--max-speed", type=float, default=10.0)
    parser.add_argument(
        "--random-start", action="store_true", help="randomize solo training starts"
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="offline")
    parser.add_argument("--wandb-project", default="crashlearn")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    args = parser.parse_args()
    if min(args.timesteps, args.n_envs, args.checkpoint_every, args.torch_threads) < 1:
        parser.error("budgets, environment count and thread count must be positive")
    if args.resume and not args.resume.is_file():
        parser.error(f"no checkpoint at {args.resume}")
    if args.initialize_from and not args.initialize_from.is_file():
        parser.error(f"no checkpoint at {args.initialize_from}")
    if not 0 < args.max_speed <= 10:
        parser.error("--max-speed must be in (0, 10]")
    if args.fixed_map_workers and args.n_envs % len(train_tracks()):
        parser.error("--fixed-map-workers needs n-envs to be a multiple of training track count")
    try:
        manifest, manifest_path = prepare_run(args)
    except ValueError as error:
        parser.error(str(error))
    manifest.setdefault(
        "config",
        {
            **PPO_CONFIG,
            "n_envs": args.n_envs,
            "device": args.device,
            "fixed_map_workers": args.fixed_map_workers,
            "max_speed": args.max_speed,
            "random_start": args.random_start,
        },
    )
    write_manifest(manifest_path, manifest)

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}; resume from the last periodic checkpoint")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        train(args, manifest, manifest_path)
    except (Exception, KeyboardInterrupt):
        if manifest["status"] == "running":
            manifest |= {"status": "failed", "error": traceback.format_exc()}
            manifest["attempts"][-1] |= {"status": "failed", "error": manifest["error"]}
            write_manifest(manifest_path, manifest)
        raise
    print(f"{manifest['driver']} seed {args.seed}: {manifest['steps_per_s']} steps/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
