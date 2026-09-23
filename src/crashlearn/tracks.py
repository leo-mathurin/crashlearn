"""Track inventory and fixed evaluation splits (single source: tracks.yaml).

The sealed test split is never returned by a default accessor: use
`sealed_test_tracks(unseal=True)` once, after the freeze (see docs/tracks.md).
"""

import hashlib
from functools import cache
from pathlib import Path

import numpy as np
import yaml

from crashlearn import SIM_DIR, add_simulator_to_path

TRACKS_YAML = Path(__file__).with_name("tracks.yaml")
MAPS_DIR = SIM_DIR / "maps"
NON_TRACK_DIRS = {"examples", "__pycache__"}
ASSET_SUFFIXES = {
    "centerline": "_centerline.csv",
    "raceline": "_raceline.csv",
    "map_image": "_map.png",
    "map_yaml": "_map.yaml",
    "donkeysim_waypoints": "_DonkeySim_waypoints.txt",
}


@cache
def _load() -> dict:
    with TRACKS_YAML.open() as f:
        return yaml.safe_load(f)


def _split(name: str) -> tuple[str, ...]:
    return tuple(sorted(t["name"] for t in _load()["tracks"] if t["split"] == name))


def train_tracks() -> tuple[str, ...]:
    return _split("train")


def validation_tracks() -> tuple[str, ...]:
    return _split("validation")


def sealed_test_tracks(*, unseal: bool = False) -> tuple[str, ...]:
    """Sealed test split: opened once after the 2027-01-17 freeze, never tuned on afterwards."""
    if not unseal:
        raise PermissionError(
            "sealed test split: pass unseal=True only for the single post-freeze evaluation"
        )
    return _split("test")


def final_retraining_tracks() -> tuple[str, ...]:
    """All distinct tracks, test included. Only for the final retraining after the freeze."""
    return tuple(sorted(t["name"] for t in _load()["tracks"]))


def canonical_name(name: str) -> str:
    for t in _load()["tracks"]:
        if name == t["name"] or name in t.get("aliases", []):
            return t["name"]
    raise KeyError(f"unknown track {name!r}")


def track_dirs() -> list[str]:
    """Directory names the simulator exposes as maps (one per track since E-12)."""
    return sorted(d.name for d in MAPS_DIR.iterdir() if d.is_dir() and d.name not in NON_TRACK_DIRS)


def inventory(track_dir: Path) -> dict[str, dict[str, str]]:
    """Asset kind -> {file, sha256} for the files found in one map directory."""
    assets = {}
    for f in sorted(track_dir.iterdir()):
        kind = next((k for k, s in ASSET_SUFFIXES.items() if f.name.endswith(s)), None)
        if kind is None:
            raise ValueError(f"unexpected file {f}")
        assets[kind] = {"file": f.name, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    return assets


def read_centerline(path: Path) -> np.ndarray:
    """(N, 2) x/y in metres of the 1:10 scale model; widths columns ignored."""
    return np.loadtxt(path, delimiter=",", comments="#", usecols=(0, 1))


def describe_geometry(xy: np.ndarray, tight_curvature: float) -> dict[str, float | int]:
    """Geometric descriptors of a closed centerline. Not a measured difficulty.

    Curvature at vertex i = |heading change| / mean length of the two adjacent segments.
    # ponytail: raw discrete curvature (~0.3 m spacing), no smoothing; resample if noisy.
    """
    seg = np.roll(xy, -1, axis=0) - xy  # closed loop: last point joins the first
    seg = seg[np.linalg.norm(seg, axis=1) > 0]
    length = np.linalg.norm(seg, axis=1)
    heading = np.arctan2(seg[:, 1], seg[:, 0])
    turn = np.abs(np.angle(np.exp(1j * (heading - np.roll(heading, 1)))))
    curvature = turn / ((length + np.roll(length, 1)) / 2)

    tight = curvature >= tight_curvature
    # number of circular runs of tight vertices
    tight_turns = int(np.sum(tight & ~np.roll(tight, 1))) or int(tight.all())
    p50, p90 = np.percentile(curvature, [50, 90])
    return {
        "geometry_length_m": round(float(length.sum()), 3),
        "geometry_total_turning_rad": round(float(turn.sum()), 3),
        "geometry_curvature_abs_p50_radpm": round(float(p50), 4),
        "geometry_curvature_abs_p90_radpm": round(float(p90), 4),
        "geometry_curvature_abs_max_radpm": round(float(curvature.max()), 4),
        "geometry_tight_turns": tight_turns,
    }


def compute_generated_fields(track_dir: Path) -> dict:
    """Fields of tracks.yaml derived from the vendored files (assets, geometry_*)."""
    assets = inventory(track_dir)
    xy = read_centerline(track_dir / assets["centerline"]["file"])
    return {"assets": assets, **describe_geometry(xy, _load()["tight_turn_curvature_radpm"])}


def load_and_step(name: str, steps: int = 20) -> dict:
    """Really load `name` in the simulator and run `steps` decisions (20 Hz) straight ahead."""
    add_simulator_to_path()
    import env_simulation as sim

    sim.set_map(name)
    sim.reset(num_cars=1)
    for _ in range(steps):
        sim.apply_action(0, 2.0, 0.0)
        sim.simulation_step()
    info = sim.get_step_info()
    return {
        "time": info["time_elapsed"],
        "progress": float(sim.get_obs(0)["progress"]),
        "status": int(info["agent_status"][0]),
        "wall": bool(info["collisions"]["wall"][0]),
    }
