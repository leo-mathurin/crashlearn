"""Modified tracks derived from the official maps (E-60), `vendor/` untouched.

A variant is named `<Base>@<kind>[+<kind>...]`, for instance `Sochi@mirror` or
`Montreal@mirror+reverse`. Kinds apply left to right:

- `reverse`: the same corridor, driven the other way. The start stays on the first point.
- `mirror`: the track flipped left to right, so left turns become right turns.
- `narrow`: both half-widths times `NARROW_FACTOR`; the walls are redrawn (`render`).

A map is three files: an occupancy image (thin black walls on a free background), a YAML
(resolution, origin) and the centerline (`x_m, y_m, w_tr_right_m, w_tr_left_m`). The
simulator reads the centerline for the start pose, the progress and the off-track check,
and the image for the LiDAR and the wall contacts. Every variant rewrites the three of
them consistently.

Variants are generated on first use under `VARIANTS_DIR` (git-ignored, reproducible), and
`install()` teaches `env_simulation.set_map` to load them. The sealed test tracks never
serve as a base: a variant of Spa would already be a look at Spa.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy import ndimage

from crashlearn import add_simulator_to_path, tracks

VARIANTS_DIR = Path(__file__).resolve().parents[2] / "runs" / "track_variants"
GENERATOR_VERSION = 1
KINDS = ("reverse", "mirror", "narrow")
NARROW_FACTOR = 0.75
WALL_PX = 3  # thickness of the official walls, measured on every map
FREE, WALL = 255, 0
# a pocket of wall-enclosed free space smaller than this inside the corridor is an
# artefact of a self-intersecting cross-section at a tight inner corner, filled back
MAX_ARTEFACT_M2 = 2.0
CENTERLINE_HEADER = "# x_m, y_m, w_tr_right_m, w_tr_left_m"


@dataclass
class TrackMap:
    """The three files of a map, in memory. `image` is the file as stored (row 0 on top)."""

    centerline: np.ndarray  # (n, 4): x, y, w_right, w_left, open polyline
    image: np.ndarray  # (h, w) uint8
    resolution: float
    origin: tuple[float, float, float]

    def pixel(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(row, col) floats in the stored image, the convention of measure_widths.py."""
        col = (x - self.origin[0]) / self.resolution
        row = self.image.shape[0] - 1 - (y - self.origin[1]) / self.resolution
        return row, col


def parse_name(name: str) -> tuple[str, tuple[str, ...]]:
    """`"Sochi@mirror+reverse"` -> `("Sochi", ("mirror", "reverse"))`."""
    base, sep, spec = name.partition("@")
    if not sep or not spec:
        raise ValueError(f"not a variant name: {name!r}")
    kinds = tuple(spec.split("+"))
    unknown = [k for k in kinds if k not in KINDS]
    if unknown:
        raise ValueError(f"unknown variant kinds {unknown} in {name!r}, known: {KINDS}")
    return tracks.canonical_name(base), kinds


def is_variant(name: str) -> bool:
    return "@" in name


def load_official(base: str) -> TrackMap:
    base = tracks.canonical_name(base)
    if base in tracks.sealed_test_tracks(unseal=True):
        raise PermissionError(f"{base} is a sealed test track: never a variant base")
    folder = tracks.MAPS_DIR / base
    meta = yaml.safe_load((folder / f"{base}_map.yaml").read_text())
    centerline = np.loadtxt(folder / f"{base}_centerline.csv", delimiter=",", comments="#")
    image = np.array(Image.open(folder / meta["image"]).convert("L"))
    return TrackMap(centerline, image, float(meta["resolution"]), tuple(meta["origin"]))


# --- transformations -----------------------------------------------------------------


def reverse(track: TrackMap) -> TrackMap:
    """Drive the other way: same first point, the others backwards, sides swapped."""
    c = track.centerline
    order = np.r_[0, np.arange(len(c) - 1, 0, -1)]
    reversed_ = c[order][:, [0, 1, 3, 2]]
    return TrackMap(reversed_, track.image.copy(), track.resolution, track.origin)


def mirror(track: TrackMap) -> TrackMap:
    """Flip x. Pixel columns flip exactly and the origin moves so cell centres match."""
    c = track.centerline.copy()
    c[:, 0] = -c[:, 0]
    c = c[:, [0, 1, 3, 2]]  # what was on the right of the car is now on its left
    width_m = track.image.shape[1] * track.resolution
    ox, oy, yaw = track.origin
    return TrackMap(c, track.image[:, ::-1].copy(), track.resolution, (-(ox + width_m), oy, yaw))


def narrow(track: TrackMap, factor: float = NARROW_FACTOR) -> TrackMap:
    c = track.centerline.copy()
    c[:, 2:4] *= factor
    redrawn = TrackMap(c, track.image, track.resolution, track.origin)
    return TrackMap(c, render(redrawn), track.resolution, track.origin)


TRANSFORMS = {"reverse": reverse, "mirror": mirror, "narrow": narrow}


# --- drawing -------------------------------------------------------------------------


def right_normals(xy: np.ndarray) -> np.ndarray:
    """Unit normal to the right of the travel direction, closed loop (measure_widths.py)."""
    d = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
    norm = np.hypot(d[:, 0], d[:, 1])
    norm[norm == 0.0] = 1.0
    return np.stack([d[:, 1], -d[:, 0]], axis=1) / norm[:, None]


def corridor_mask(track: TrackMap) -> np.ndarray:
    """True where the corridor described by the centerline and its widths lies."""
    c = track.centerline
    n = right_normals(c[:, :2])
    right = c[:, :2] + n * c[:, 2:3]
    left = c[:, :2] - n * c[:, 3:4]
    h, w = track.image.shape
    canvas = Image.new("1", (w, h), 0)
    draw = ImageDraw.Draw(canvas)

    def px(points: np.ndarray) -> list[tuple[float, float]]:
        row, col = track.pixel(points[:, 0], points[:, 1])
        return list(zip(col.tolist(), row.tolist(), strict=True))

    lp, rp, cp = px(left), px(right), px(c[:, :2])
    count = len(c)
    for i in range(count):
        j = (i + 1) % count
        # two half-quads per section: centerline to each side, never a bowtie across it
        draw.polygon([cp[i], cp[j], rp[j], rp[i]], fill=1)
        draw.polygon([cp[i], cp[j], lp[j], lp[i]], fill=1)
    return np.array(canvas, dtype=bool)


def render(track: TrackMap) -> np.ndarray:
    """Occupancy image of the corridor: free inside, walls `WALL_PX` thick around it."""
    corridor = corridor_mask(track)
    outside, count = ndimage.label(~corridor)
    if count:
        sizes = ndimage.sum_labels(np.ones_like(outside), outside, range(1, count + 1))
        small = np.flatnonzero(sizes * track.resolution**2 < MAX_ARTEFACT_M2) + 1
        corridor |= np.isin(outside, small)
    distance = ndimage.distance_transform_edt(~corridor)
    image = np.full(corridor.shape, FREE, dtype=np.uint8)
    image[(distance > 0) & (distance <= WALL_PX)] = WALL
    return image


# --- files ---------------------------------------------------------------------------


def build(name: str) -> TrackMap:
    base, kinds = parse_name(name)
    track = load_official(base)
    for kind in kinds:
        track = TRANSFORMS[kind](track)
    return track


def folder(name: str, root: Path = VARIANTS_DIR) -> Path:
    return root / "maps" / name


def write(name: str, track: TrackMap, root: Path = VARIANTS_DIR) -> Path:
    """Write the map atomically: parallel workers may ask for the same variant."""
    target = folder(name, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=target.parent))
    try:
        Image.fromarray(track.image).save(staging / f"{name}_map.png")
        meta = {
            "image": f"{name}_map.png",
            "resolution": track.resolution,
            "origin": [float(v) for v in track.origin],
            "negate": 0,
            "occupied_thresh": 0.45,
            "free_thresh": 0.196,
        }
        (staging / f"{name}_map.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
        np.savetxt(
            staging / f"{name}_centerline.csv",
            track.centerline,
            delimiter=", ",
            fmt="%.6f",
            header=CENTERLINE_HEADER[2:],
        )
        provenance = {"generator_version": GENERATOR_VERSION, "name": name}
        (staging / "variant.yaml").write_text(yaml.safe_dump(provenance))
        try:
            os.rename(staging, target)
        except OSError:
            if not target.is_dir():  # someone else won the race otherwise
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def ensure(name: str, root: Path = VARIANTS_DIR) -> Path:
    """Generate the variant if missing or produced by an older generator."""
    target = folder(name, root)
    stamp = target / "variant.yaml"
    if stamp.exists():
        if yaml.safe_load(stamp.read_text())["generator_version"] == GENERATOR_VERSION:
            return target
    if target.exists():
        shutil.rmtree(target)
    return write(name, build(name), root)


# --- simulator hook ------------------------------------------------------------------


def install(root: Path = VARIANTS_DIR) -> None:
    """Let `env_simulation.set_map` load variants. Calling it again changes nothing.

    `set_map` builds `<_REPO_ROOT>/maps/<name>/` paths once, then keeps absolute paths: we
    point `_REPO_ROOT` at the variants root for the duration of that single call.
    """
    add_simulator_to_path()
    import env_simulation as sim

    if getattr(sim.set_map, "variants", False):
        return
    official_set_map = sim.set_map

    def set_map(name: str, num_cars: int | None = None) -> None:
        if not is_variant(name):
            return official_set_map(name, num_cars)
        if name == sim._current_map:
            return None
        ensure(name, root)
        available = sim._ensure_maps_discovered()
        repo_root = sim._REPO_ROOT
        available.add(name)
        sim._REPO_ROOT = str(root)
        try:
            return official_set_map(name, num_cars)
        finally:
            sim._REPO_ROOT = repo_root
            available.discard(name)

    set_map.variants = True
    sim.set_map = set_map
