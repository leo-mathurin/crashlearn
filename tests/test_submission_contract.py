"""Tournament contract imposed by the subject for submission/<driver>/.

Fixed by the subject: `Agent()` loads its model relative to its own file,
`Agent.predict(obs, info) -> (float, float)` with bounded values, and agent.py
imports only numpy, onnxruntime and the standard library.

Naming assumption (not fixed by the subject): an agent detects a new episode
when `info["step_count"] == 0`.
"""

import ast
import math
import sys

import numpy as np
import pytest
from agent_loader import create_dummy_info, create_dummy_obs, validate_and_load_agent
from conftest import SUBMISSION_DIRS, require_model

ALLOWED_MODULES = {"numpy", "onnxruntime"} | set(sys.stdlib_module_names)
SPEED_BOUNDS = (-2.0, 10.0)
STEER_LIMIT = 0.4189 + 1e-4  # engine s_max is 0.4189 rad, rounded
ACTION_ATOL = 1e-6

submissions = pytest.mark.parametrize(
    "submission_dir",
    SUBMISSION_DIRS
    or [pytest.param(None, marks=pytest.mark.skip(reason="no submission/<driver>/agent.py yet"))],
    ids=lambda p: p.name if p else "none",
)


def forbidden_imports(source: str) -> list[str]:
    """Imports outside numpy, onnxruntime and the stdlib. Static only: __import__ is not seen."""
    bad = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: agent.py must be self-contained
                bad.append("." * node.level + (node.module or ""))
                continue
            names = [node.module]
        else:
            continue
        bad += [name for name in names if name.split(".")[0] not in ALLOWED_MODULES]
    return bad


def check_action(result) -> None:
    assert isinstance(result, tuple) and len(result) == 2, f"expected a 2-tuple, got {result!r}"
    speed, steer = result
    assert type(speed) is float and type(steer) is float, (
        f"expected Python floats, got {type(speed).__name__}, {type(steer).__name__}"
    )
    assert math.isfinite(speed) and math.isfinite(steer)
    assert SPEED_BOUNDS[0] <= speed <= SPEED_BOUNDS[1], f"target_speed out of bounds: {speed}"
    assert abs(steer) <= STEER_LIMIT, f"steering out of bounds: {steer}"


def load_agent(submission_dir, unique_id):
    require_model(submission_dir)
    return validate_and_load_agent(submission_dir, unique_id=unique_id)[1]


# --- checker self-tests (always active) ------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import numpy as np\nimport onnxruntime as ort\nimport os, math", []),
        ("from pathlib import Path\nfrom numpy.typing import NDArray", []),
        ("import torch", ["torch"]),
        ("from stable_baselines3 import PPO", ["stable_baselines3"]),
        ("def f():\n    import gymnasium", ["gymnasium"]),
        ("from . import features", ["."]),
        ("from .features import build", [".features"]),
    ],
)
def test_forbidden_imports_detection(source, expected):
    assert forbidden_imports(source) == expected


@pytest.mark.parametrize("result", [(0.0, 0.0), (10.0, 0.4189), (-2.0, -0.4189)])
def test_check_action_accepts_valid(result):
    check_action(result)


@pytest.mark.parametrize(
    "result",
    [
        [1.0, 0.0],
        (1.0,),
        (np.float32(1.0), 0.0),
        (1.0, np.float64(0.0)),
        (1, 0.0),
        (float("nan"), 0.0),
        (10.5, 0.0),
        (-2.5, 0.0),
        (1.0, 0.5),
    ],
)
def test_check_action_rejects_invalid(result):
    with pytest.raises(AssertionError):
        check_action(result)


# --- real submissions (skipped until submission/<driver>/ exists) -----------


@submissions
def test_agent_imports_only_allowed_modules(submission_dir):
    source = (submission_dir / "agent.py").read_text(encoding="utf-8")
    assert forbidden_imports(source) == []


@submissions
def test_delivery_files_present(submission_dir):
    for name in ("agent.py", "model.onnx", "model.onnx.data"):
        assert (submission_dir / name).is_file(), f"missing {submission_dir.name}/{name}"


@submissions
def test_agent_loads_from_arbitrary_cwd(submission_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = load_agent(submission_dir, 0)  # the official loader dry-runs the loader schema
    check_action(agent.predict(create_dummy_obs(), create_dummy_info()))


@submissions
def test_agent_predicts_on_simulator_observation(submission_dir, sim):
    agent = load_agent(submission_dir, 0)
    for car in (0, 1):
        check_action(agent.predict(sim.get_obs(car), sim.get_step_info()))


@submissions
def test_agent_state_resets_between_episodes(submission_dir, sim):
    obs, info = sim.get_obs(0), sim.get_step_info()
    episode_start = {**info, "step_count": 0}
    expected = load_agent(submission_dir, 0).predict(obs, episode_start)

    agent = load_agent(submission_dir, 1)
    for k in range(20):
        lidar = np.clip(obs["lidar"] * (0.5 + k / 20), 0.1, 15.0).astype(np.float32)
        agent.predict({**obs, "lidar": lidar}, {**info, "step_count": k})
    assert agent.predict(obs, episode_start) == pytest.approx(expected, abs=ACTION_ATOL)


def test_agents_from_different_folders_do_not_interfere(sim):
    if len(SUBMISSION_DIRS) < 2:
        pytest.skip("needs at least two submission/<driver>/ folders")
    for d in SUBMISSION_DIRS:
        require_model(d)
    obs, info = sim.get_obs(0), sim.get_step_info()
    alone = [load_agent(d, i).predict(obs, info) for i, d in enumerate(SUBMISSION_DIRS)]
    # load every agent first, then predict: catches shared module state or "last model wins"
    together = [load_agent(d, 100 + i) for i, d in enumerate(SUBMISSION_DIRS)]
    for agent, expected in zip(together, alone, strict=True):
        assert agent.predict(obs, info) == pytest.approx(expected, abs=ACTION_ATOL)
