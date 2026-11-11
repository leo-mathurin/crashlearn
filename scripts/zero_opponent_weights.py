"""Remove the untrained opponent inputs of a solo driver before it races other cars.

A solo driver only sees empty opponent slots (features 105 to 125 of each frame) while it
trains, so their first-layer weights keep their random initialization. With opponents on
track, these weights add an offset that was never learned. This derivation sets them to zero
in the actor and the critic: with opponents, the driver acts exactly as alone on track. No
training is added and the source run is preserved; the manifest records the transformation.
"""

import argparse
import uuid
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from crashlearn import add_simulator_to_path
from crashlearn.experiments import (
    load_completed_run,
    now,
    provenance,
    require_empty,
    save_model,
    sha256,
    write_manifest,
)
from crashlearn.features import EGO_SIZE, FEATURE_SIZE

add_simulator_to_path()  # SB3 saves read the version of the simulator's `gym` namespace


def opponent_columns(depth: int) -> list[int]:
    """The opponent features of every stacked frame, newest or oldest alike."""
    return [
        frame * FEATURE_SIZE + column
        for frame in range(depth)
        for column in range(EGO_SIZE, FEATURE_SIZE)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.run_dir / "manifest.json"
    try:
        manifest = load_completed_run(args.run_dir)
        require_empty(args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    current = provenance()
    depth = manifest["history_depth"]
    columns = opponent_columns(depth)

    torch.set_num_threads(1)
    model = PPO.load(args.run_dir / "model.zip", device="cpu")
    if model.observation_space.shape != (FEATURE_SIZE * depth,):
        parser.error("model observation shape differs from the manifest")
    inputs = np.random.default_rng(0).uniform(-1, 1, (256, FEATURE_SIZE * depth)).astype(np.float32)
    masked = inputs.copy()
    masked[:, columns] = 0
    expected = model.predict(masked, deterministic=True)[0]
    with torch.no_grad():
        for network in (
            model.policy.mlp_extractor.policy_net,
            model.policy.mlp_extractor.value_net,
        ):
            network[0].weight[:, columns] = 0
    actual = model.predict(inputs, deterministic=True)[0]
    np.testing.assert_array_equal(actual, expected)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_sha256 = save_model(model, args.output_dir / "model.zip")
    (args.output_dir / "source-manifest.json").write_bytes(source.read_bytes())
    derived = {
        **manifest,
        "run_id": uuid.uuid4().hex,
        "created_utc": now(),
        "attempts": [],
        "model_sha256": model_sha256,
        "derivation": {
            "method": "zero_opponent_weights",
            "feature_columns": columns,
            "source_run_id": manifest["run_id"],
            "source_model_sha256": manifest["model_sha256"],
            "source_manifest_sha256": sha256(source),
            # an earlier derivation of the source, kept whole: the chain stays auditable
            "source_derivation": manifest.get("derivation"),
            "provenance": current,
            "parity_samples": len(inputs),
            "max_normalized_action_error": float(np.max(np.abs(actual - expected))),
            "training_steps_added": 0,
        },
    }
    for key in ("wall_time_s", "steps_per_s", "total_wall_time_s"):
        derived.pop(key, None)
    write_manifest(args.output_dir / "manifest.json", derived)
    print(f"Derived {args.output_dir}; source preserved at {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
