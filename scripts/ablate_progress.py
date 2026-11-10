"""Create a reactive inference model with no connections from lap/progress features.

The completed training run is preserved. The derived manifest records its source and
the exact transformation; this command does not perform any further training.
"""

import argparse
import json
import uuid
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from crashlearn.env import action_to_controls, controls_to_action
from crashlearn.evaluation import provenance, sha256
from crashlearn.experiments import now, write_manifest
from crashlearn.features import FEATURE_SIZE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-speed", type=float, help="optional lower inference speed limit, m/s")
    args = parser.parse_args()
    source = args.run_dir / "manifest.json"
    manifest = json.loads(source.read_text())
    model_path = args.run_dir / "model.zip"
    if manifest["status"] != "completed" or sha256(model_path) != manifest["model_sha256"]:
        parser.error("source run must be completed and match its model hash")
    current = provenance()
    for key in ("features_sha256", "history_sha256"):
        if manifest[key] != current[key]:
            parser.error(f"source run has a different {key}")
    if manifest["history_depth"] != 1 or manifest.get("derivation"):
        parser.error("source must be an unmodified reactive training run")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory is not empty")

    torch.set_num_threads(1)
    model = PPO.load(model_path, device="cpu")
    if model.observation_space.shape != (FEATURE_SIZE,):
        parser.error("model observation shape differs from the manifest")
    source_high = model.action_space.high.copy()
    if args.max_speed is not None:
        if not 0 < args.max_speed <= action_to_controls(source_high)[0]:
            parser.error("max-speed must be positive and cannot exceed the source action limit")
        upper = controls_to_action(args.max_speed, 0.0)[0]
    else:
        upper = source_high[0]
    inputs = np.random.default_rng(0).uniform(-1, 1, (256, FEATURE_SIZE)).astype(np.float32)
    masked = inputs.copy()
    masked[:, 102:105] = 0
    expected = model.predict(masked, deterministic=True)[0]
    expected[:, 0] = np.minimum(expected[:, 0], upper)
    with torch.no_grad():
        for network in (
            model.policy.mlp_extractor.policy_net,
            model.policy.mlp_extractor.value_net,
        ):
            network[0].weight[:, 102:105] = 0
    model.action_space.high[0] = upper
    model.policy.action_space.high[0] = upper
    actual = model.predict(inputs, deterministic=True)[0]
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The vendored gym namespace has no version, which SB3's save metadata reads.
    import gym

    if not hasattr(gym, "__version__"):
        gym.__version__ = "vendored-namespace"
    model.save(args.output_dir / "model.zip")
    (args.output_dir / "source-manifest.json").write_bytes(source.read_bytes())
    derived = {
        **manifest,
        "run_id": uuid.uuid4().hex,
        "created_utc": now(),
        "attempts": [],
        "model_sha256": sha256(args.output_dir / "model.zip"),
        "derivation": {
            "method": "zero_progress_weights",
            "feature_columns": [102, 103, 104],
            "source_run_id": manifest["run_id"],
            "source_model_sha256": manifest["model_sha256"],
            "source_manifest_sha256": sha256(source),
            "provenance": current,
            "parity_samples": len(inputs),
            "max_normalized_action_error": float(np.max(np.abs(actual - expected))),
            "training_steps_added": 0,
            "requested_max_speed_mps": args.max_speed,
            "source_action_high": source_high.tolist(),
            "derived_action_high": model.action_space.high.tolist(),
        },
    }
    for key in ("wall_time_s", "steps_per_s", "total_wall_time_s"):
        derived.pop(key, None)
    write_manifest(args.output_dir / "manifest.json", derived)
    print(f"Derived {args.output_dir}; source preserved at {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
