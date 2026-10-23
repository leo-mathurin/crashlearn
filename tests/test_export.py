"""ONNX export contract (E-17). Skipped until torch and crashlearn.export are available.

torch is not installed in CI (train group): when E-17 lands, add a CPU-only job
that runs `uv sync --locked --group train` and this file.

Naming assumption (NOT fixed by the subject): `crashlearn.export.export_onnx(policy, path,
sample_input)` exports a `torch.nn.Module` mapping a float32 feature batch to actions,
writing `path` (and `path.data` when weights are external).
"""

import numpy as np
import onnxruntime as ort
from conftest import import_if_present

torch = import_if_present("torch")
export = import_if_present("crashlearn.export")

TORCH_ONNX_ATOL = 1e-5
N_FEATURES = 16


def test_onnx_matches_torch(tmp_path):
    torch.manual_seed(0)
    policy = torch.nn.Sequential(
        torch.nn.Linear(N_FEATURES, 32), torch.nn.Tanh(), torch.nn.Linear(32, 2)
    ).eval()
    inputs = np.random.default_rng(0).uniform(-1, 1, (64, N_FEATURES)).astype(np.float32)
    path = tmp_path / "model.onnx"

    export.export_onnx(policy, path, torch.from_numpy(inputs[:1]))

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (onnx_out,) = session.run(None, {session.get_inputs()[0].name: inputs})
    with torch.no_grad():
        torch_out = policy(torch.from_numpy(inputs)).numpy()
    np.testing.assert_allclose(onnx_out, torch_out, atol=TORCH_ONNX_ATOL, rtol=0)
