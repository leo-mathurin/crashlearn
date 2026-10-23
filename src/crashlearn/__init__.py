"""Crash & Learn application package."""

import sys
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parents[2] / "vendor" / "simulation"


def add_simulator_to_path() -> None:
    """Make the vendored `env_simulation` module importable."""
    if str(SIM_DIR) not in sys.path:
        sys.path.insert(0, str(SIM_DIR))
