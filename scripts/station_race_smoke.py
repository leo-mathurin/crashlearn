"""Exercise the E-14 race wrapper, periodic checkpoints, and resume on the station."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import gym
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from crashlearn.env import CrashLearnEnv


def _checkpoint(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--map", default="example")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--resume-steps", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=256)
    args = parser.parse_args()
    if min(args.steps, args.resume_steps, args.checkpoint_every) < 1:
        parser.error("step counts must be positive")
    if args.steps % args.checkpoint_every:
        parser.error("--steps must be a multiple of --checkpoint-every")
    if args.checkpoint_every % 128 or args.resume_steps % 128:
        parser.error("checkpoint and resume intervals must be multiples of 128")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    # The vendored simulator creates an empty gym namespace. SB3's save metadata
    # reads gym.__version__, although training itself uses Gymnasium.
    if not hasattr(gym, "__version__"):
        gym.__version__ = "vendored-namespace"

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = CrashLearnEnv(num_cars=1, maps=args.map, max_episode_steps=256)
    try:
        model = PPO(
            "MlpPolicy",
            env,
            device=args.device,
            seed=42,
            n_steps=128,
            batch_size=64,
            n_epochs=1,
            verbose=0,
        )
        periodic = CheckpointCallback(
            save_freq=args.checkpoint_every,
            save_path=str(output),
            name_prefix="ppo-race",
        )
        started = time.perf_counter()
        model.learn(total_timesteps=args.steps, callback=periodic)
        elapsed = time.perf_counter() - started
        if model.num_timesteps != args.steps:
            raise RuntimeError(f"unexpected training length: {model.num_timesteps}")
        checkpoints = sorted(
            output.glob("ppo-race_*_steps.zip"), key=lambda path: int(path.stem.split("_")[1])
        )
        expected_steps = list(range(args.checkpoint_every, args.steps + 1, args.checkpoint_every))
        if [int(path.stem.split("_")[1]) for path in checkpoints] != expected_steps:
            raise RuntimeError("periodic checkpoints do not match the policy")

        # The callback runs before PPO updates at a rollout boundary. Save the
        # post-update model as the exact state from which to resume this run.
        final = output / f"ppo-race-final-{model.num_timesteps}.zip"
        model.save(final)
        observation = model.get_env().reset()
        expected_action, _ = model.predict(observation, deterministic=True)
        restored = PPO.load(final, env=env, device=args.device)
        actual_action, _ = restored.predict(observation, deterministic=True)
        if not np.allclose(actual_action, expected_action):
            raise RuntimeError("checkpoint changed the deterministic action")
        restored.learn(total_timesteps=args.resume_steps, reset_num_timesteps=False)
        if restored.num_timesteps != args.steps + args.resume_steps:
            raise RuntimeError(f"unexpected resumed length: {restored.num_timesteps}")
        resumed = output / f"ppo-race-resumed-{restored.num_timesteps}.zip"
        restored.save(resumed)

        # An interrupted run restarts from its last periodic checkpoint, not from
        # a final save: check that this file restores and resumes too.
        periodic_restored = PPO.load(checkpoints[-1], env=env, device=args.device)
        if periodic_restored.num_timesteps != expected_steps[-1]:
            raise RuntimeError(f"periodic checkpoint restored at {periodic_restored.num_timesteps}")
        periodic_restored.learn(total_timesteps=args.resume_steps, reset_num_timesteps=False)
        if periodic_restored.num_timesteps != expected_steps[-1] + args.resume_steps:
            raise RuntimeError(
                f"unexpected periodic resumed length: {periodic_restored.num_timesteps}"
            )

        result = {
            "map": args.map,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "torch": torch.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "checkpoint_every_steps": args.checkpoint_every,
            "training_steps": model.num_timesteps,
            "resumed_steps": restored.num_timesteps,
            "periodic_resumed_steps": periodic_restored.num_timesteps,
            "training_seconds": round(elapsed, 3),
            "checkpoint_reload_equal": True,
            "checkpoints": [_checkpoint(path) for path in checkpoints],
            "final_checkpoint": _checkpoint(final),
            "resumed_checkpoint": _checkpoint(resumed),
        }
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
