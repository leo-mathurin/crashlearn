"""Train a PPO driver with a persistent manifest, checkpoints, CSV and optional W&B.

Depth 1 is the reactive baseline; depth 4 has memory. Resume into the original output
folder with a larger target budget. Each attempt keeps its own logs. Environment RNG and
partial rollouts are not restored, so a resumed run is not an exact replay.
"""

import argparse
import signal
import sys
from pathlib import Path

from crashlearn import map_cache
from crashlearn.env import SPEED_MAX, CrashLearnEnv
from crashlearn.experiments import Run, allow_sb3_saves, attach_wandb, checkpoint_step, sha256
from crashlearn.history_env import FrictionWrapper, HistoryWrapper
from crashlearn.tracks import train_tracks
from crashlearn.weather import WeatherConfig

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


def make_env(depth: int, track: str | None, config: dict):
    def build():
        map_cache.install()
        weather = config["weather"]
        env = HistoryWrapper(
            CrashLearnEnv(
                maps=track,
                max_speed=config["max_speed"],
                random_start=config["random_start"],
                weather=WeatherConfig(**weather) if weather else None,
            ),
            depth,
        )
        return FrictionWrapper(env) if config["friction_feature"] else env

    return build


def copy_policy(source, model) -> None:
    """Copy compatible weights. A model with one input more (friction) gets a zero column,
    so it starts by driving exactly as its source."""
    import torch

    state = source.policy.state_dict()
    target = model.policy.state_dict()
    for name, value in state.items():
        if value.shape != target[name].shape:
            rows, columns = value.shape
            if target[name].shape != (rows, columns + 1):
                raise ValueError("initialization checkpoint has a different observation shape")
            state[name] = torch.cat([value, torch.zeros(rows, 1, dtype=value.dtype)], dim=1)
    model.policy.load_state_dict(state)


def build_model(args: argparse.Namespace, env, run: Run):
    from stable_baselines3 import PPO

    if args.resume:
        return PPO.load(args.resume, env=env, device=args.device)
    model = PPO("MlpPolicy", env, seed=args.seed, device=args.device, **PPO_CONFIG)
    if args.initialize_from:
        source = PPO.load(args.initialize_from, device=args.device)
        copy_policy(source, model)
        run.manifest["initialization"] = {
            "checkpoint": str(args.initialize_from),
            "checkpoint_sha256": sha256(args.initialize_from),
            "source_timesteps": source.num_timesteps,
            "optimizer_restored": False,
        }
        model.set_random_seed(args.seed)
    return model


def train(args: argparse.Namespace, run: Run) -> None:
    with run.attempt(
        budget_timesteps=args.timesteps,
        resumed_from=args.resume,
        device=args.device,
        torch_threads=args.torch_threads,
    ) as cleanups:
        import torch
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.logger import configure
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

        allow_sb3_saves()
        torch.set_num_threads(args.torch_threads)
        maps = train_tracks()
        builds = [
            make_env(
                args.history_depth,
                maps[i % len(maps)] if args.fixed_map_workers else None,
                run.manifest["config"],
            )
            for i in range(args.n_envs)
        ]
        env = VecMonitor(SubprocVecEnv(builds))
        cleanups.callback(env.close)
        model = build_model(args, env, run)
        logger = configure(str(run.log_dir), ["stdout", "csv"])
        cleanups.callback(logger.close)
        writer = None
        if args.wandb:
            writer = attach_wandb(
                logger,
                run,
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.wandb_group,
                mode=args.wandb_mode,
            )
        model.set_logger(logger)
        run.attach(model)

        class TrackedCheckpoint(CheckpointCallback):
            def _on_step(self):
                result = super()._on_step()
                if self.n_calls % self.save_freq == 0:
                    path = Path(self._checkpoint_path(extension="zip"))
                    run.record_checkpoint(path)
                    if writer is not None:
                        writer.checkpoint(path)
                return result

        checkpoints = TrackedCheckpoint(
            save_freq=max(args.checkpoint_every // args.n_envs, 1),
            save_path=str(run.directory / "checkpoints"),
            name_prefix=f"ppo_attempt{run.number:03d}",
        )
        model.learn(
            total_timesteps=args.timesteps - model.num_timesteps,
            callback=checkpoints,
            reset_num_timesteps=False,
        )
        final = run.complete()
        if writer is not None:
            writer.checkpoint(final)


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
    parser.add_argument("--max-speed", type=float, default=SPEED_MAX)
    parser.add_argument(
        "--random-start", action="store_true", help="randomize solo training starts"
    )
    parser.add_argument(
        "--weather", action="store_true", help="tournament weather that changes in the race"
    )
    parser.add_argument(
        "--nominal-share", type=float, default=0.25, help="with --weather: nominal episodes"
    )
    parser.add_argument(
        "--friction-feature",
        action="store_true",
        help="E-23 measurement: append friction_current to the observation (not exportable)",
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
    if not 0 < args.max_speed <= SPEED_MAX:
        parser.error(f"--max-speed must be in (0, {SPEED_MAX}]")
    if not 0 <= args.nominal_share <= 1:
        parser.error("--nominal-share must be in [0, 1]")
    if args.fixed_map_workers and args.n_envs % len(train_tracks()):
        parser.error("--fixed-map-workers needs n-envs to be a multiple of training track count")
    config = {
        **PPO_CONFIG,
        "n_envs": args.n_envs,
        "device": args.device,
        "fixed_map_workers": args.fixed_map_workers,
        "max_speed": args.max_speed,
        "random_start": args.random_start,
        "weather": WeatherConfig(nominal_share=args.nominal_share).to_dict()
        if args.weather
        else None,
        "friction_feature": args.friction_feature,
    }
    try:
        run = Run.open(
            args.output_dir,
            history_depth=args.history_depth,
            seed=args.seed,
            config=config,
            resume=args.resume,
        )
        # checked before the attempt, so a refused resume leaves a completed run usable
        if args.resume and args.timesteps <= checkpoint_step(args.resume):
            raise ValueError("target --timesteps must exceed the checkpoint step")
    except ValueError as error:
        parser.error(str(error))

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}; resume from the last periodic checkpoint")

    signal.signal(signal.SIGTERM, interrupt)
    train(args, run)
    print(f"{run.manifest['driver']} seed {args.seed}: {run.manifest['steps_per_s']} steps/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
