"""Crash & Learn application package."""

import sys
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parents[2] / "vendor" / "simulation"
# Decisions (20 Hz) before a race ends as TIMEOUT, in training, evaluation and confirmation.
MAX_DECISIONS = 8000


def engine_seed(rng: np.random.Generator) -> int:
    """Simulator seed of one episode, drawn from its episode RNG.

    `CrashLearnEnv.reset(seed=s, options={"map": ...})` draws it first, so a race seed `s`
    gives `engine_seed(np.random.default_rng(s))` without Gymnasium.
    """
    return int(rng.integers(2**31))


def add_simulator_to_path() -> None:
    """Make the vendored `env_simulation` module importable."""
    if str(SIM_DIR) not in sys.path:
        sys.path.insert(0, str(SIM_DIR))
