import importlib
import importlib.util
from pathlib import Path

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
