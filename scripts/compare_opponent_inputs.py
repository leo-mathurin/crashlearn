"""Check that a driver without opponent weights acts as its source alone on track.

Random features with opponents go to the new model; the same features with every opponent
column at zero go to the old one. Both ONNX models must give the same actions. The second
number is the offset that the untrained weights of the old model add with opponents.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from crashlearn.features import EGO_SIZE, FEATURE_SIZE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", type=Path, required=True, help="submission folder")
    parser.add_argument("--old", type=Path, required=True, help="submission folder")
    parser.add_argument("--samples", type=int, default=4096)
    args = parser.parse_args()
    depth = json.loads((args.new / "export.json").read_text())["history_depth"]
    new, old = (
        ort.InferenceSession(str(folder / "model.onnx"), providers=["CPUExecutionProvider"])
        for folder in (args.new, args.old)
    )
    rng = np.random.default_rng(7)
    inputs = rng.uniform(-1, 1, (args.samples, depth * FEATURE_SIZE)).astype(np.float32)
    alone = inputs.reshape(args.samples, depth, FEATURE_SIZE).copy()
    alone[:, :, EGO_SIZE:] = 0
    alone = alone.reshape(args.samples, -1)
    reference = old.run(None, {"features": alone})[0]
    result = {
        "samples": args.samples,
        "history_depth": depth,
        "max_abs_new_vs_old_alone": float(
            np.abs(new.run(None, {"features": inputs})[0] - reference).max()
        ),
        "max_abs_old_with_vs_without_opponents": float(
            np.abs(old.run(None, {"features": inputs})[0] - reference).max()
        ),
    }
    print(json.dumps(result, indent=2))
    return 0 if result["max_abs_new_vs_old_alone"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
