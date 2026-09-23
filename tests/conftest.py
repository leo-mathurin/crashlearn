import importlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from crashlearn import add_simulator_to_path

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIRS = sorted(
    p for p in (ROOT / "submission").glob("*") if p.is_dir() and (p / "agent.py").is_file()
)

# Vendored simulator modules (env_simulation, agent_loader) importable from every test.
add_simulator_to_path()


def import_if_present(name: str):
    """Skip the calling test module until `name` exists.

    Unlike pytest.importorskip, an ImportError raised *inside* an existing module still fails.
    """
    if importlib.util.find_spec(name) is None:
        pytest.skip(f"{name} is not available yet", allow_module_level=True)
    return importlib.import_module(name)


def require_model(submission_dir: Path) -> None:
    """Skip if model.onnx is missing or an un-fetched Git LFS pointer (CI uses lfs: false)."""
    model = submission_dir / "model.onnx"
    if not model.is_file():
        pytest.skip(f"no model.onnx in {submission_dir.name}")
    with model.open("rb") as f:
        if f.read(24).startswith(b"version https://git-lfs"):
            pytest.skip(f"{submission_dir.name}/model.onnx is a Git LFS pointer")


@pytest.fixture(scope="session")
def sim():
    """Simulator with 2 cars, stepped once. Session-scoped: the first step costs ~10 s of JIT."""
    import env_simulation

    env_simulation.reset(num_cars=2)
    env_simulation.simulation_step()
    yield env_simulation
    env_simulation.close()


def pure_pursuit(sim_instance, cid: int, speed: float = 3.0, lookahead: float = 1.0):
    """Minimal scripted pure-pursuit driver (mirrors `sim_recorder._pure_pursuit_policy`),
    used to make a car complete laps deterministically in dynamics tests.

    `sim_instance` is the private `_Sim` singleton (`env_simulation._get_sim()`); the steering
    bounds live on the `env_simulation` module itself, not on the singleton.
    """
    import env_simulation

    st = sim_instance._sim.agents[cid].state
    x, y, yaw = float(st[0]), float(st[1]), float(st[4])
    w = sim_instance._waypoints
    idx = int(np.argmin((w[:, 0] - x) ** 2 + (w[:, 1] - y) ** 2))
    n = len(w)
    cum = 0.0
    target = idx
    for i in range(1, n):
        wi = (idx + i) % n
        wj = (idx + i - 1) % n
        cum += math.hypot(w[wi, 0] - w[wj, 0], w[wi, 1] - w[wj, 1])
        if cum >= lookahead:
            target = wi
            break
    angle = math.atan2(w[target, 1] - y, w[target, 0] - x) - yaw
    angle = math.atan2(math.sin(angle), math.cos(angle))
    params = env_simulation._PARAMS
    return speed, max(params["s_min"], min(params["s_max"], angle))
