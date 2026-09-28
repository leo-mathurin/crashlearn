"""Check CUDA computation, short PPO training, and checkpoint restore."""

import argparse
import json
import time
from pathlib import Path

import onnx
import onnxruntime
import stable_baselines3
import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def matrix_benchmark(device: str, size: int, repetitions: int) -> dict[str, float]:
    generator = torch.Generator().manual_seed(42)
    left_cpu = torch.randn(size, size, generator=generator)
    right_cpu = torch.randn(size, size, generator=generator)
    left = left_cpu.to(device)
    right = right_cpu.to(device)

    # The same input matrices and precision are used on both devices.
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = torch.mm(left_cpu, right_cpu)
    actual = torch.mm(left, right)
    synchronize(device)
    torch.testing.assert_close(actual.cpu(), reference, rtol=1e-3, atol=1e-3)

    for _ in range(3):
        torch.mm(left, right)
    synchronize(device)

    start = time.perf_counter()
    for _ in range(repetitions):
        torch.mm(left, right)
    synchronize(device)
    seconds = time.perf_counter() - start
    return {
        "matrix_size": size,
        "repetitions": repetitions,
        "elapsed_seconds": round(seconds, 6),
        "gflops": round((2 * size**3 * repetitions) / seconds / 1e9, 3),
    }


def train_and_restore(device: str, output_dir: Path, timesteps: int) -> dict[str, object]:
    environment = make_vec_env("CartPole-v1", n_envs=2, seed=42)
    model = PPO(
        "MlpPolicy",
        environment,
        device=device,
        seed=42,
        n_steps=64,
        batch_size=64,
        n_epochs=1,
        verbose=0,
    )
    start = time.perf_counter()
    model.learn(total_timesteps=timesteps)
    seconds = time.perf_counter() - start

    checkpoint = output_dir / f"ppo-cartpole-{device}"
    observation = environment.reset()
    expected_actions, _ = model.predict(observation, deterministic=True)
    model.save(str(checkpoint))
    restored = PPO.load(str(checkpoint), env=environment, device=device)
    actions, _ = restored.predict(observation, deterministic=True)
    assert len(actions) == 2
    assert (actions == expected_actions).all()
    environment.close()

    return {
        "environment": "CartPole-v1",
        "timesteps": model.num_timesteps,
        "elapsed_seconds": round(seconds, 6),
        "steps_per_second": round(model.num_timesteps / seconds, 3),
        "checkpoint": str(checkpoint.with_suffix(".zip")),
        "checkpoint_bytes": checkpoint.with_suffix(".zip").stat().st_size,
        "restored_prediction": actions.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrix-size", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=512)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "versions": {
            "python": __import__("sys").version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "stable_baselines3": stable_baselines3.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
            "wandb": wandb.__version__,
        },
        "matrix": matrix_benchmark(args.device, args.matrix_size, args.repetitions),
        "training": train_and_restore(args.device, args.output_dir, args.timesteps),
    }
    output = args.output_dir / f"station-smoke-{args.device}.json"
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
