"""agent_loader.py — Dynamic loading and validation of agent submissions.

Loads one or more team submission directories, validates file structures,
imports modules in isolated namespaces, and verifies prediction contract.
"""

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def create_dummy_obs() -> Dict[str, Any]:
    """Create a valid dummy observation dict conforming to get_obs() contract."""
    return {
        "agent_id": 0,
        "lidar": np.full((100,), 5.0, dtype=np.float32),
        "velocity": 0.0,
        "steering": 0.0,
        "progress": 0.0,
        "lap_count": 0,
        "rank": 1,
        "opponents": {
            0: {"x_rel": 0.0, "y_rel": 0.0, "yaw_rel": 0.0, "speed": 0.0,
                "progress": 0.0, "lap_count": 0, "status": 0},
            1: {"x_rel": 0.0, "y_rel": 0.0, "yaw_rel": 0.0, "speed": 0.0,
                "progress": 0.0, "lap_count": 0, "status": 0},
            2: {"x_rel": 0.0, "y_rel": 0.0, "yaw_rel": 0.0, "speed": 0.0,
                "progress": 0.0, "lap_count": 0, "status": 0},
            3: {"x_rel": 0.0, "y_rel": 0.0, "yaw_rel": 0.0, "speed": 0.0,
                "progress": 0.0, "lap_count": 0, "status": 0},
        },
    }


def create_dummy_info() -> Dict[str, Any]:
    """Create a valid dummy info dict conforming to get_step_info() contract."""
    return {
        "step_count": 0,
        "time_elapsed": 0.0,
        "ranks": {0: 1, 1: 2, 2: 3, 3: 4},
        "collisions": {
            "wall":    {i: False for i in range(4)},
            "vehicle": {i: False for i in range(4)},
        },
        "progress_delta": {i: 0.0 for i in range(4)},
        "lap_complete": {i: False for i in range(4)},
        "stagnation": {i: False for i in range(4)},
        "friction_current": 1.0,
        "opponents_mask": np.array([True, False, False, False]),
        "agent_status": {0: 1, 1: 0, 2: 0, 3: 0},
        "lap_times": {i: [] for i in range(4)},
        "race_over": False,
        "max_progress": {i: 0.0 for i in range(4)},
    }


def sanitize_team_name(name: str, max_len: int = 10) -> str:
    """Sanitize and clamp team name to max_len characters."""
    clean = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip()
    if not clean:
        clean = "team"
    return clean[:max_len]


def validate_and_load_agent(team_dir: str | Path, team_name: Optional[str] = None, unique_id: int = 0) -> Tuple[str, Any]:
    """Validate and dynamically load an Agent from a submission directory.

    Args:
        team_dir: Path to submission directory containing agent.py and model.onnx.
        team_name: Display name for the team (defaults to directory name).
        unique_id: Integer suffix for unique module naming in sys.modules.

    Returns:
        Tuple of (sanitized_team_name, agent_instance).

    Raises:
        ValueError / FileNotFoundError / RuntimeError on validation failure with precise diagnostics.
    """
    path = Path(team_dir).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"[ERROR] [Team: {team_name or path.name}] Directory does not exist: {path}")

    raw_name = team_name if team_name else path.name
    clean_name = sanitize_team_name(raw_name)

    agent_py = path / "agent.py"
    model_onnx = path / "model.onnx"

    if not agent_py.is_file():
        raise FileNotFoundError(f"[ERROR] [Team: {clean_name}] Missing required 'agent.py' in {path}")
    if not model_onnx.is_file():
        raise FileNotFoundError(f"[ERROR] [Team: {clean_name}] Missing required 'model.onnx' in {path}")

    # Dynamic isolated module loading
    module_name = f"submission_{clean_name}_{unique_id}"
    spec = importlib.util.spec_from_file_location(module_name, str(agent_py))
    if spec is None or spec.loader is None:
        raise ImportError(f"[ERROR] [Team: {clean_name}] Could not create module spec for {agent_py}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"[ERROR] [Team: {clean_name}] Failed to execute {agent_py}: {e}") from e

    if not hasattr(module, "Agent"):
        raise AttributeError(f"[ERROR] [Team: {clean_name}] 'agent.py' does not define an 'Agent' class.")

    # Instantiation check
    try:
        agent_instance = module.Agent()
    except Exception as e:
        raise RuntimeError(f"[ERROR] [Team: {clean_name}] Failed to instantiate Agent(): {e}") from e

    if not hasattr(agent_instance, "predict"):
        raise AttributeError(f"[ERROR] [Team: {clean_name}] Agent instance missing 'predict(obs, info)' method.")

    # Dry-run inference check
    dummy_obs = create_dummy_obs()
    dummy_info = create_dummy_info()
    try:
        res = agent_instance.predict(dummy_obs, dummy_info)
    except Exception as e:
        raise RuntimeError(f"[ERROR] [Team: {clean_name}] Agent.predict() crashed during validation dry-run: {e}") from e

    if not isinstance(res, (tuple, list)) or len(res) != 2:
        raise ValueError(
            f"[ERROR] [Team: {clean_name}] Agent.predict() must return a tuple (speed, steering), got {type(res)}: {res}"
        )

    speed, steer = res
    if type(speed) is not float or type(steer) is not float:
        raise TypeError(
            f"[ERROR] [Team: {clean_name}] Agent.predict() return values must be pure Python float scalars. "
            f"Got speed={type(speed).__name__}, steer={type(steer).__name__}. Cast with float(target_speed), float(steering)."
        )

    return clean_name, agent_instance


def discover_and_load_submissions(
    submissions_path: str | Path,
    max_teams: int = 4,
) -> List[Tuple[str, Any]]:
    """Discover and load up to max_teams submissions from a directory or path.

    Supports:
    - Single submission directory containing agent.py + model.onnx.
    - Multi-team parent directory containing subfolders (team_1/, team_2/, etc.).

    Returns:
        List of (team_name, agent_instance).
    """
    root = Path(submissions_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"[ERROR] Submissions path does not exist: {root}")

    # Discover team subdirectories first (grand-prix mode takes priority)
    subdirs = sorted([d for d in root.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))])

    if subdirs:
        # Multi-team directory: validate count and load
        if len(subdirs) < max_teams:
            raise ValueError(
                f"[ERROR] Grand Prix mode requires at least {max_teams} team subdirectories, found {len(subdirs)} in: {root}\n"
                f"Expected structure: <submission_dir>/<team_name>/{{'agent.py', 'model.onnx'}}"
            )

        loaded_teams: List[Tuple[str, Any]] = []
        errors: List[str] = []

        for idx, team_dir in enumerate(subdirs[:max_teams]):
            try:
                name, agent = validate_and_load_agent(team_dir, team_name=team_dir.name, unique_id=idx)
                loaded_teams.append((name, agent))
            except Exception as e:
                errors.append(f"Team '{team_dir.name}': {e}")

        if errors:
            error_msg = "\n".join(errors)
            raise RuntimeError(f"Failed to load team submissions from {root}:\n{error_msg}")

        return loaded_teams

    # No subdirectories: fallback to flat agent (self-play, only for max_teams == 1)
    if max_teams == 1 and (root / "agent.py").is_file() and (root / "model.onnx").is_file():
        team_name, agent = validate_and_load_agent(root, team_name=root.name, unique_id=0)
        return [(team_name, agent)]

    # No subdirs and no flat agent, or multi-car with no teams
    if subdirs:
        raise ValueError(f"[ERROR] No valid team subdirectories found in: {root}")

    raise RuntimeError(
        f"[ERROR] Multi-team mode requires team subdirectories (e.g. team_alpha/, team_bravo/...) or a flat submission/ with agent.py + model.onnx for single-car.\n"
        f"Expected structure: <submission_dir>/<team_name>/{{'agent.py', 'model.onnx'}}"
    )
