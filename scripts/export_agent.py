"""Export one completed PPO run to a tournament agent and check numerical parity."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from stable_baselines3 import PPO

from crashlearn.env import action_to_controls
from crashlearn.experiments import load_completed_run, provenance, require_empty, sha256
from crashlearn.export import DeterministicPPO, agent_source, export_onnx
from crashlearn.features import FEATURE_SIZE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--friction-exponent",
        type=float,
        help="E-49: bound the speed at the model limit times friction_current ** exponent",
    )
    args = parser.parse_args()
    if args.friction_exponent is not None and args.friction_exponent < 0:
        parser.error("--friction-exponent must be non-negative")
    try:
        manifest = load_completed_run(args.run_dir)
        require_empty(args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    current = provenance()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = PPO.load(args.run_dir / "model.zip", device="cpu")
    depth = manifest["history_depth"]
    if model.observation_space.shape != (FEATURE_SIZE * depth,):
        parser.error("model observation shape differs from the manifest")
    inputs = np.random.default_rng(0).uniform(-1, 1, (256, FEATURE_SIZE * depth)).astype(np.float32)
    path = args.output_dir / "model.onnx"
    export_onnx(DeterministicPPO(model.policy), path, torch.from_numpy(inputs[:1]))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"features": inputs})[0]
    expected = model.predict(inputs, deterministic=True)[0]
    np.testing.assert_allclose(outputs, expected, atol=1e-5, rtol=0)
    friction_cap = None
    if args.friction_exponent is not None:
        reference = action_to_controls(model.action_space.high)[0]
        friction_cap = {"reference_mps": reference, "exponent": args.friction_exponent}
    (args.output_dir / "agent.py").write_text(agent_source(depth, friction_cap))
    result = {
        **current,
        "source_run_id": manifest["run_id"],
        "source_model_sha256": manifest["model_sha256"],
        "derivation": manifest.get("derivation"),
        "history_depth": depth,
        "action_low": model.action_space.low.tolist(),
        "action_high": model.action_space.high.tolist(),
        "max_normalized_action_error": float(np.max(np.abs(outputs - expected))),
        "parity_samples": len(inputs),
        "parity_atol": 1e-5,
        "opset": 17,
        "friction_speed_cap": friction_cap,
        "files": {p.name: sha256(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()},
    }
    (args.output_dir / "export.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Exported {args.output_dir}; "
        f"max normalized action error {result['max_normalized_action_error']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
