"""ONNX export contract (E-25), run by the Training and ONNX CPU job.

Naming assumption (NOT fixed by the subject): `crashlearn.export.export_onnx(policy, path,
sample_input)` exports a `torch.nn.Module` mapping a float32 feature batch to actions,
writing `path` (and `path.data` when weights are external).
"""

import numpy as np
import onnxruntime as ort
import pytest
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


def test_export_keeps_the_trained_action_bounds(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    from crashlearn.env import CrashLearnEnv

    env = CrashLearnEnv(max_speed=5.0)
    try:
        model = sb3.PPO("MlpPolicy", env, device="cpu", seed=0)
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.copy_(torch.tensor([5.0, -5.0]))
        inputs = np.zeros((4, 126), dtype=np.float32)
        expected = model.predict(inputs, deterministic=True)[0]
        path = tmp_path / "bounded.onnx"
        export.export_onnx(
            export.DeterministicPPO(model.policy), path, torch.from_numpy(inputs[:1])
        )
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        (actual,) = session.run(None, {session.get_inputs()[0].name: inputs})
        np.testing.assert_allclose(actual, expected, atol=TORCH_ONNX_ATOL, rtol=0)
        np.testing.assert_allclose(actual[:, 0], 1 / 6, atol=TORCH_ONNX_ATOL, rtol=0)
    finally:
        env.close()
