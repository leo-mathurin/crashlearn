"""Modified tracks derived from the official maps (E-60), `vendor/` untouched.

A variant is named `<Base>@<kind>[+<kind>...]`, for instance `Sochi@mirror` or
`Montreal@mirror+reverse`. Kinds apply left to right:

- `reverse`: the same corridor, driven the other way. The start stays on the first point.
- `mirror`: the track flipped left to right, so left turns become right turns.
- `narrow`: both half-widths times `NARROW_FACTOR`; the walls are redrawn (`render`).
- `bend<seed>`: one bend tightened or widened, drawn at random from the seed.
- `section<seed>`: one stretch pushed sideways by a smooth bump, drawn from the seed.

`bend` and `section` move the centerline only inside a window, so the loop stays closed and
the start untouched. A draw is kept only if the new corridor stays drivable (`check`);
otherwise the next draw of the same seed is tried. `Sochi@bend3` is always the same track.

A real track that the school left out of its archive, `Norisring`, is built the same way
from `new_tracks/` (`load_new`) and accepts the same kinds (`Norisring@mirror`).

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
import re
import shutil
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

from crashlearn import add_simulator_to_path, tracks

VARIANTS_DIR = Path(__file__).resolve().parents[2] / "runs" / "track_variants"
NEW_TRACKS_DIR = Path(__file__).with_name("new_tracks")
# Suzuka, the other track left out, crosses itself (a bridge): the simulator projects the
# car on the nearest centerline point, so the progress jumps by 41 % of a lap at the
# crossing and the car is declared DNF. It cannot be driven in this simulator.
NEW_TRACKS = ("Norisring",)
# How the school scaled the TUMFTM centerlines, median over the 19 train and validation
# maps: real length times 0.0801, corridor 2.229 m wide (its own measurement), the track
# spread over 1726 pixels of a 2000 x 2000 image, centred.
SCHOOL_SCALE = 0.0801
# Norisring's straights run 25.8 m apart for real: at 0.0801 the two 2.229 m corridors
# merge. 0.128 is the smallest scale that keeps 3.3 m between their centerlines (corridor,
# two walls, GAP_CLEARANCE_M), which gives 294 m a lap, close to IMS and Oschersleben.
NEW_TRACK_SCALE = {"Norisring": 0.128}
SCHOOL_WIDTH_M = 2.229
SCHOOL_EXTENT_PX = 1726
IMAGE_PX = 2000
GENERATOR_VERSION = 2
KINDS = ("reverse", "mirror", "narrow", "bend<seed>", "section<seed>")
KIND_PATTERN = re.compile(r"(reverse|mirror|narrow|(bend|section)(\d+))")
NARROW_FACTOR = 0.75
# geometric edits (bend, section)
SMOOTH_M = 6.0  # curvature is averaged over this length: below it, it is drawing noise
BEND_MIN_CURVATURE = 0.15  # rad/m, radius < 6.7 m: what counts as a bend
TIGHTEN = (0.55, 0.85)  # bend radius scale when tightened
WIDEN = (1.3, 2.0)  # bend radius scale when widened
SECTION_LENGTH_M = (6.0, 16.0)
SECTION_SHIFT_M = (0.6, 1.8)
START_GUARD_M = 15.0  # the start and the grid behind it are never edited
MIN_RADIUS_M = 1.0  # smoothed, at the centerline; the car turns on 0.74 m at full lock
INNER_CLEARANCE_M = 0.3  # inner edge radius left once the inner half-width is taken
GAP_CLEARANCE_M = 0.5  # between two corridors that the edit brought closer
MAX_DRAWS = 50
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
    unknown = [k for k in kinds if not KIND_PATTERN.fullmatch(k)]
    if unknown:
        raise ValueError(f"unknown variant kinds {unknown} in {name!r}, known: {KINDS}")
    return base_name(base), kinds


def base_name(base: str) -> str:
    return base if base in NEW_TRACKS else tracks.canonical_name(base)


def is_variant(name: str) -> bool:
    return "@" in name


def is_generated(name: str) -> bool:
    """A name `install()` serves from `VARIANTS_DIR`: a variant or a new track."""
    return is_variant(name) or name in NEW_TRACKS


def load_new(name: str) -> TrackMap:
    """A TUMFTM track the school left out, at the scale and width of its maps."""
    real = np.loadtxt(NEW_TRACKS_DIR / f"{name}.csv", delimiter=",", comments="#")
    xy = (real[:, :2] - real[0, :2]) * NEW_TRACK_SCALE[name]  # the start on (0, 0), like the others
    half = np.full((len(xy), 2), SCHOOL_WIDTH_M / 2)
    centerline = np.hstack([xy, half])
    resolution = float(np.ptp(xy, axis=0).max() / SCHOOL_EXTENT_PX)
    centre = (xy.min(axis=0) + xy.max(axis=0)) / 2
    origin = (*(centre - IMAGE_PX * resolution / 2), 0.0)
    blank = np.full((IMAGE_PX, IMAGE_PX), FREE, dtype=np.uint8)
    return _redrawn(TrackMap(centerline, blank, resolution, origin))


def load_base(base: str) -> TrackMap:
    return load_new(base) if base in NEW_TRACKS else load_official(base)


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


# --- geometric edits -----------------------------------------------------------------


def _loop_geometry(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arc length at each point, step to the next one, smoothed signed curvature (+ = left)."""
    step = np.hypot(*(np.roll(xy, -1, axis=0) - xy).T)
    heading = np.arctan2(*(np.roll(xy, -1, axis=0) - xy).T[::-1])
    turn = np.angle(np.exp(1j * (heading - np.roll(heading, 1))))
    curvature = turn / np.maximum((step + np.roll(step, 1)) / 2, 1e-9)
    window = max(3, int(round(SMOOTH_M / step.mean())) | 1)
    smooth = ndimage.uniform_filter1d(curvature, window, mode="wrap")
    return np.r_[0.0, np.cumsum(step)[:-1]], step, smooth


def _editable(arc: np.ndarray, total: float) -> np.ndarray:
    return (arc > START_GUARD_M) & (arc < total - START_GUARD_M)


def find_bends(track: TrackMap) -> list[tuple[int, int, int]]:
    """(first, apex, last) index of each bend, sharpest first, away from the start."""
    arc, step, curvature = _loop_geometry(track.centerline[:, :2])
    editable = _editable(arc, arc[-1] + step[-1])
    strength = np.where(editable, np.abs(curvature), 0.0)
    bends = []
    while strength.max() >= BEND_MIN_CURVATURE:
        apex = int(np.argmax(strength))
        inside = np.abs(curvature) >= strength[apex] / 2
        first = last = apex
        while first > 0 and inside[first - 1] and editable[first - 1]:
            first -= 1
        while last < len(arc) - 1 and inside[last + 1] and editable[last + 1]:
            last += 1
        bends.append((first, apex, last))
        strength[max(0, first - 1) : last + 2] = 0.0
    return bends


def _taper(arc: np.ndarray, start: float, plateau: tuple[float, float], end: float) -> np.ndarray:
    """0 outside [start, end], 1 on the plateau, raised-cosine ramps in between."""
    weight = np.zeros_like(arc)
    up = (arc > start) & (arc < plateau[0])
    weight[up] = (1 - np.cos(np.pi * (arc[up] - start) / (plateau[0] - start))) / 2
    weight[(arc >= plateau[0]) & (arc <= plateau[1])] = 1.0
    down = (arc > plateau[1]) & (arc < end)
    weight[down] = (1 + np.cos(np.pi * (arc[down] - plateau[1]) / (end - plateau[1]))) / 2
    return weight


def scale_bend(track: TrackMap, bend: tuple[int, int, int], scale: float) -> TrackMap:
    """Scale the bend radius about its centre of curvature, blended in and out."""
    xy = track.centerline[:, :2]
    arc, _, curvature = _loop_geometry(xy)
    first, apex, last = bend
    radius = 1.0 / curvature[apex]  # signed: > 0 for a left turn, centre on the left
    left = -right_normals(xy)[apex]
    centre = xy[apex] + left * radius
    ramp = max(SMOOTH_M, arc[last] - arc[first])
    lo = max(arc[first] - ramp, START_GUARD_M)
    hi = min(arc[last] + ramp, arc[-1] - START_GUARD_M)
    weight = _taper(arc, lo, (arc[first], arc[last]), hi)
    moved = weight > 0  # elsewhere, leave the points bit for bit
    factor = 1.0 + (scale - 1.0) * weight[moved]
    edited = track.centerline.copy()
    edited[moved, :2] = centre + (xy[moved] - centre) * factor[:, None]
    return TrackMap(edited, track.image, track.resolution, track.origin)


def shift_section(track: TrackMap, start_m: float, length_m: float, shift_m: float) -> TrackMap:
    """Push a stretch sideways (+ = right) by a smooth bump that is zero at both ends."""
    xy = track.centerline[:, :2]
    arc, _, _ = _loop_geometry(xy)
    u = (arc - start_m) / length_m
    bump = np.where((u > 0) & (u < 1), (1 - np.cos(2 * np.pi * u)) / 2, 0.0)
    edited = track.centerline.copy()
    edited[:, :2] = xy + right_normals(xy) * (shift_m * bump)[:, None]
    return TrackMap(edited, track.image, track.resolution, track.origin)


def check(edited: TrackMap, original: TrackMap) -> str | None:
    """Why the edited corridor is not drivable, or None."""
    xy = edited.centerline[:, :2]
    _, step, curvature = _loop_geometry(xy)
    radius = 1.0 / np.maximum(np.abs(curvature), 1e-9)
    if radius.min() < MIN_RADIUS_M:
        return f"radius {radius.min():.2f} m < {MIN_RADIUS_M} m"
    inner = np.where(curvature > 0, edited.centerline[:, 3], edited.centerline[:, 2])
    if np.any(radius - inner < INNER_CLEARANCE_M):
        return "the inner edge folds over itself"
    row, col = edited.pixel(xy[:, 0], xy[:, 1])
    margin = 2 * WALL_PX + 2
    h, w = edited.image.shape
    if row.min() < margin or col.min() < margin or row.max() > h - margin or col.max() > w - margin:
        return "the track leaves the image"
    # two parts of the loop that the edit brought closer than their corridors allow
    half = edited.centerline[:, 2:].max(axis=1)
    reach = 2 * half.max() + 2 * WALL_PX * edited.resolution + GAP_CLEARANCE_M
    arc = np.r_[0.0, np.cumsum(step)[:-1]]
    total = arc[-1] + step[-1]
    before = original.centerline[:, :2]
    pairs = cKDTree(xy).query_pairs(reach, output_type="ndarray")
    if len(pairs):
        i, j = pairs.T
        along = np.abs(arc[i] - arc[j])
        far = np.minimum(along, total - along) >= 3 * reach  # not neighbours on one stretch
        gap = np.hypot(*(xy[i] - xy[j]).T)
        need = half[i] + half[j] + 2 * WALL_PX * edited.resolution + GAP_CLEARANCE_M
        closer = gap < np.hypot(*(before[i] - before[j]).T) - 1e-6
        bad = far & (gap < need) & closer
        if bad.any():
            k = int(np.argmax(bad))
            return f"two stretches {gap[k]:.2f} m apart, {need[k]:.2f} m needed"
    return None


def _rng(base: str, kind: str, seed: int) -> np.random.Generator:
    return np.random.default_rng([zlib.crc32(f"{base}/{kind}".encode()), seed])


def random_bend(track: TrackMap, base: str, seed: int) -> TrackMap:
    bends = find_bends(track)
    if not bends:
        raise ValueError(f"{base}: no bend sharper than {BEND_MIN_CURVATURE} rad/m")
    rng = _rng(base, "bend", seed)
    for _ in range(MAX_DRAWS):
        bend = bends[rng.integers(len(bends))]
        scale = rng.uniform(*(TIGHTEN if rng.random() < 0.5 else WIDEN))
        edited = scale_bend(track, bend, scale)
        if check(edited, track) is None:
            return _redrawn(edited)
    raise ValueError(f"{base}@bend{seed}: no drivable draw in {MAX_DRAWS}")


def random_section(track: TrackMap, base: str, seed: int) -> TrackMap:
    arc, step, _ = _loop_geometry(track.centerline[:, :2])
    total = arc[-1] + step[-1]
    rng = _rng(base, "section", seed)
    for _ in range(MAX_DRAWS):
        length = rng.uniform(*SECTION_LENGTH_M)
        start = rng.uniform(START_GUARD_M, total - START_GUARD_M - length)
        shift = rng.uniform(*SECTION_SHIFT_M) * rng.choice([-1.0, 1.0])
        edited = shift_section(track, start, length, shift)
        if check(edited, track) is None:
            return _redrawn(edited)
    raise ValueError(f"{base}@section{seed}: no drivable draw in {MAX_DRAWS}")


def _redrawn(track: TrackMap) -> TrackMap:
    return TrackMap(track.centerline, render(track), track.resolution, track.origin)


TRANSFORMS = {"reverse": reverse, "mirror": mirror, "narrow": narrow}


def apply(track: TrackMap, kind: str, base: str) -> TrackMap:
    match = KIND_PATTERN.fullmatch(kind)
    if match is None:
        raise ValueError(f"unknown variant kind {kind!r}")
    if match[2] == "bend":
        return random_bend(track, base, int(match[3]))
    if match[2] == "section":
        return random_section(track, base, int(match[3]))
    return TRANSFORMS[kind](track)


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
    if name in NEW_TRACKS:
        return load_new(name)
    base, kinds = parse_name(name)
    track = load_base(base)
    for kind in kinds:
        track = apply(track, kind, base)
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
        if not is_generated(name):
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
