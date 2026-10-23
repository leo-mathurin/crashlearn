"""
env_simulation.py — Crash&Learn Grand Prix simulation backend.

Private _Sim singleton wrapping f110_gym.envs.base_classes.Simulator directly
(NOT F110Env, NOT gym.make). Exposes exactly 7 module-level functions.

LiDAR class-variable trap: RaceCar.scan_simulator is a class-level singleton.
After every Simulator (re)construction _apply_lidar_override() must be called to
build the 100-beam scanner and its geometry arrays. Multiple Simulator instances
in ONE interpreter share these class variables (harmless under SubprocVecEnv process
isolation; relevant for the eval harness and unit tests that build simulators in sequence).

Wall response: the engine's iTTC STOP does state[3:]=0 on wall contact, which
clobbers yaw (state[4]) to 0 — a latent engine bug that snaps the heading to
+x and allows the car to escape. The wrapper neutralises this with a per-sub-
step FULL-POSE ROLLBACK (restoring x, y, AND yaw from a pre-step snapshot on
wall contact). Velocity/yaw_rate/slip remain zeroed (correct for a stopped car).
No speed debuff. No reward penalty. Escape from a head-on wall requires
reversing (single-track model produces no yaw at v=0).
Collisions are never penalised and never terminate the episode.
"""

# ---------------------------------------------------------------------------
# Repo path — f110_gym package lives one level up from this file's parent
# ---------------------------------------------------------------------------
import os
import warnings

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_GYM_PATH = os.path.join(_REPO_ROOT, "gym")
if _GYM_PATH not in __import__("sys").path:
    __import__("sys").path.insert(0, _GYM_PATH)

warnings.filterwarnings("ignore", message="Chosen integrator is RK4")

# ---------------------------------------------------------------------------
# Real imports (after bootstrap)
# ---------------------------------------------------------------------------
import math
import collections
import numpy as np
from numba import njit

from f110_gym.envs.base_classes import Simulator, Integrator, RaceCar
from f110_gym.envs.laser_models import ScanSimulator2D, xy_2_rc

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_FREQ_HZ           = 100         # physics sub-step rate (Hz)
_DECISION_FREQ_HZ  = 20          # control / bookkeeping rate (Hz)
_PHYSICS_STEPS     = 5           # sub-steps per decision step
_FRICTION_LO       = 0.99        # minimum surface friction (engine mu) 1=sunny, 0.5=heavy rain
_FRICTION_HI       = 1.0         # maximum surface friction
_FRICTION_INTERVAL_SEC = (20, 20)  # FIXED 20 s between friction changes (matches lore)
_DNF_WINDOW_SEC    = 10.0        # stagnation detection window (s)
_DNF_MIN_GAIN_M    = 1.0         # min forward progress (m of arc) required over that window
_OFFTRACK_MARGIN_M = 1.00        # tolerance past the track edge before DNF (m)
_DEFAULT_HALF_WIDTH_M = 1.05     # fallback half-width when the CSV lacks the columns
_TARGET_SPEED_MIN  = -2.0        # wrapper action bound (engine v_min is lower)
_TARGET_SPEED_MAX  = 10.0        # wrapper action bound (engine v_max is higher)
_VEL_OBS_MIN       = -2.0        # observation contract bound for velocity
_VEL_OBS_MAX       = 10.0        # observation contract bound for velocity
_LIDAR_RAYS        = 100         # beam count (class-var override after every build)
_LIDAR_FOV         = 2.0 * math.pi  # 360-degree full field of view (rad)
_LIDAR_SCAN_FOV    = _LIDAR_FOV * (_LIDAR_RAYS - 1) / _LIDAR_RAYS  # span cut by one increment, else beam 0 and beam 99 overlap
_LIDAR_NOISE       = 0.001       # multiplicative uniform noise half-range: 0.001 (±0.1%, dry) up to 0.1 (±10%, heavy rain)
_LIDAR_MIN         = 0.1         # clip min (m)
_LIDAR_MAX         = 15.0        # clip max (m)
_TTC_THRESH        = 0.015       # iTTC detection window (1.5 x dt at 100 Hz)
_MAX_SLOTS         = 4           # hard cap: 4 cars
_REQUIRED_LAPS     = 3           # number of completed laps to be FINISHED (status=2)
_KNOCKBACK_REST    = 0.15        # restitution coefficient for vehicle-vehicle impulse
_KNOCKBACK_MAX_M   = 0.5         # cap on the displacement a single contact transfers (m)
_KNOCKBACK_MIN_SHARE = 0.2       # min share of a contact each car takes (caps the other at 0.8)
_WALL_REST         = 0.3         # wall restitution: recoil = 0.3 x impact speed
_WALL_REST_MAX     = 0.45        # recoil cap (m/s); < 0.5 or the dynamic model diverges
_DNF_WINDOW_STEPS  = int(_DNF_WINDOW_SEC * _DECISION_FREQ_HZ)   # 200 steps (10 s × 20 Hz)

# Per-car odometer lap detection constants (DESIGN.md § "Per-car odometer")
_GRID_LAT_M        = 0.6         # lateral gap between the two cars of a row (2×2 grid)
_GRID_ROW_M        = 0.8         # longitudinal gap between the two grid rows

# ---------------------------------------------------------------------------
# Default vehicle physics parameters (from F110Env defaults)
# ---------------------------------------------------------------------------
_PARAMS: dict = {
    "mu":       1.0489,
    "C_Sf":     4.718,
    "C_Sr":     5.4562,
    "lf":       0.15875,
    "lr":       0.17145,
    "h":        0.074,
    "m":        3.74,
    "I":        0.04712,
    "s_min":   -0.4189,
    "s_max":    0.4189,
    "sv_min":  -3.2,
    "sv_max":   3.2,
    "v_switch": 7.319,
    "a_max":    9.51,
    "v_min":   -5.0,
    "v_max":    20.0,
    "width":    0.31,
    "length":   0.58,
}

# ---------------------------------------------------------------------------
# Map / centerline config  (matches examples/config_example_map.yaml)
# ---------------------------------------------------------------------------
_EXAMPLES_DIR = os.path.join(_REPO_ROOT, "maps", "examples")
_MAP_YAML     = os.path.join(_EXAMPLES_DIR, "example_map.yaml")
_MAP_EXT      = ".png"
_WPT_PATH     = os.path.join(_EXAMPLES_DIR, "example_centerline.csv")

# Start pose — loaded from the config YAML that accompanies each map.
# Convention: for examples/example_map.yaml → examples/config_example_map.yaml
_MAP_CONFIG_YAML = _MAP_YAML.replace(".yaml", "_config.yaml").replace("_map.yaml", "_config.yaml")

# ---------------------------------------------------------------------------
# Map switching state (DESIGN.md § "Switching Map Circuits")
#   Students control when to switch via set_map().  The wrapper tracks what
#   is currently loaded so reset() can reuse it without per-call arguments.
# ---------------------------------------------------------------------------
_current_map: str | None = None          # name of map currently loaded (None → default/example)
_available_maps: set[str] | None = None  # cached discovery result


def _ensure_maps_discovered() -> set[str]:
    """Lazy, one-shot discovery of available maps under <repo>/maps/."""
    global _available_maps
    if _available_maps is None:
        maps_root = os.path.join(_REPO_ROOT, "maps")
        _available_maps = {
            d
            for d in os.listdir(maps_root)
            if os.path.isdir(os.path.join(maps_root, d))
            and d not in ("examples", "__pycache__")
        }
    return _available_maps
_WPT_DELIM    = ";"
_WPT_ROWSKIP  = 3
_WPT_XIND     = 1
_WPT_YIND     = 2
# Start pose loaded from config YAML (sx, sy, stheta)
try:
    import yaml as _yaml_module
    with open(_MAP_CONFIG_YAML) as _f:
        _cfg = _yaml_module.safe_load(_f)
    _MAP_SX       = float(_cfg["sx"])
    _MAP_SY       = float(_cfg["sy"])
    _MAP_STHETA   = float(_cfg["stheta"])
except Exception:
    # Fallback to hardcoded values if config YAML is missing
    _MAP_SX       = 0.7
    _MAP_SY       = 0.0
    _MAP_STHETA   = 1.37079632679

# ---------------------------------------------------------------------------
# njit progress helper — copied from examples/waypoint_follow.py (avoids
# importing that file which pulls in legacy gym/pyglet)
# ---------------------------------------------------------------------------
@njit(fastmath=False, cache=True)
def _project_progress_njit(px, py, wpts, arc, total_arc):
    N = wpts.shape[0]
    best_seg = 0
    best_t = 0.0
    best_dist_sq = 1e30
    for i in range(N - 1):
        dx = wpts[i + 1, 0] - wpts[i, 0]
        dy = wpts[i + 1, 1] - wpts[i, 1]
        l2 = dx * dx + dy * dy
        if l2 > 0.0:
            t = ((px - wpts[i, 0]) * dx + (py - wpts[i, 1]) * dy) / l2
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        else:
            t = 0.0
        projx = wpts[i, 0] + t * dx
        projy = wpts[i, 1] + t * dy
        ddx = px - projx
        ddy = py - projy
        d2 = ddx * ddx + ddy * ddy
        if d2 < best_dist_sq:
            best_dist_sq = d2
            best_seg = i
            best_t = t
    seg_len = arc[best_seg + 1] - arc[best_seg]
    return (arc[best_seg] + best_t * seg_len) / total_arc


@njit(fastmath=False, cache=True)
def _project_lateral_offset_njit(px, py, wpts):
    """Project (x,y) onto centerline → returns (lateral, best_seg).
    lateral = signed distance from centerline (positive = right side).
    best_seg = index of closest segment on centerline.
    """
    N = wpts.shape[0]
    best_seg = 0
    best_t = 0.0
    best_dist_sq = 1e30
    for i in range(N - 1):
        dx = wpts[i + 1, 0] - wpts[i, 0]
        dy = wpts[i + 1, 1] - wpts[i, 1]
        l2 = dx * dx + dy * dy
        if l2 > 0.0:
            t = ((px - wpts[i, 0]) * dx + (py - wpts[i, 1]) * dy) / l2
            t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        else:
            t = 0.0
        projx = wpts[i, 0] + t * dx
        projy = wpts[i, 1] + t * dy
        ddx = px - projx
        ddy = py - projy
        d2 = ddx * ddx + ddy * ddy
        if d2 < best_dist_sq:
            best_dist_sq = d2
            best_seg = i
            best_t = t
    # Compute lateral offset: cross product of (point - proj) with segment direction
    dx = wpts[best_seg + 1, 0] - wpts[best_seg, 0]
    dy = wpts[best_seg + 1, 1] - wpts[best_seg, 1]
    projx = wpts[best_seg, 0] + best_t * dx
    projy = wpts[best_seg, 1] + best_t * dy
    # Lateral = cross product (right-hand rule, positive = right side)
    lateral = (px - projx) * dy - (py - projy) * dx
    seg_len = np.sqrt(dx * dx + dy * dy)
    if seg_len > 0.0:
        lateral = lateral / seg_len
    return lateral, float(best_seg)


# ---------------------------------------------------------------------------
# _Sim singleton
# ---------------------------------------------------------------------------
class _Sim:
    """
    Private singleton wrapping a f110_gym Simulator instance.

    Construction contract
    ---------------------
    Simulator.__init__ takes (params, num_agents, seed, time_step, integrator).
    num_beams / fov are NOT Simulator params — they live as class variables on
    RaceCar. _apply_lidar_override() must be called after every (re)construction.
    """

    def __init__(self) -> None:
        self._sim: Simulator | None  = None
        self._num_agents: int        = 0
        self._map_name: str          = ""   # Currently loaded map name (set by set_map)
        # RNG seeds. _wrapper_seed drives the wrapper's uniform LiDAR noise, the
        # only scan noise in play: the engine's Gaussian noise never fires, since
        # scan() is called with rng=None. reset(seed=...) sets both, for replay.
        self._wrapper_seed: int      = 0
        self._engine_seed: int       = 42
        self._rng                    = np.random.default_rng(seed=self._wrapper_seed)
        self._waypoints: np.ndarray  = np.zeros((2, 2))  # (N, 2) x,y
        self._arc: np.ndarray        = np.zeros(2)        # cumulative arc length (N,)
        self._total_arc: float       = 1.0

        # Per-agent state (indexed 0..3)
        self._actions:    np.ndarray  = np.zeros((_MAX_SLOTS, 2))  # [steer, speed]
        self._steering_cmd: np.ndarray = np.zeros(_MAX_SLOTS)       # last commanded steer

        # Own-start phase storage (DESIGN.md § "Per-car odometer")
        #   _progress[i]    = rel one step ago (current lap phase for obs["progress"])
        #   _rel_prev[i]    = previous rel (used internally for d computation)
        self._progress:   np.ndarray  = np.zeros(_MAX_SLOTS)
        self._rel_prev:   np.ndarray  = np.zeros(_MAX_SLOTS)

        # Odometer state per car
        self._progress_start: np.ndarray = np.zeros(_MAX_SLOTS)   # absolute progress of start pose
        self._cum:            np.ndarray = np.zeros(_MAX_SLOTS)    # cumulative fractional laps from own start
        self._last_delta:     np.ndarray = np.zeros(_MAX_SLOTS)    # signed wrap-aware per-step d (for get_step_info)

        self._lap_count:         np.ndarray  = np.zeros(_MAX_SLOTS, dtype=int)
        self._last_lap_complete: np.ndarray  = np.zeros(_MAX_SLOTS, dtype=bool)
        self._status:            np.ndarray  = np.zeros(_MAX_SLOTS, dtype=int)  # 0=DNF,1=ACTIVE,2=FIN
        self._lap_times:    list   = [[] for _ in range(_MAX_SLOTS)]            # completed-lap times (s)
        self._lap_start_step: np.ndarray = np.zeros(_MAX_SLOTS, dtype=int)      # step the current lap began
        self._finish_step:  np.ndarray = np.full(_MAX_SLOTS, -1, dtype=int)     # step of FINISH (-1 if not)
        self._progress_history: list   = [collections.deque(maxlen=_DNF_WINDOW_STEPS + 1)
                                          for _ in range(_MAX_SLOTS)]
        self._stagnation_flag: np.ndarray = np.zeros(_MAX_SLOTS, dtype=bool)

        # Progress-based DNF: running max of the odometer _cum, in laps ∈ [0, _REQUIRED_LAPS].
        # Never negative (max seeded at 0) even if _cum goes so in reverse — that is FIX E5.
        self._max_progress: np.ndarray = np.zeros(_MAX_SLOTS)

        # Track width: (num_segments, 2) = [w_right, w_left] per segment
        self._track_widths: np.ndarray = np.full((0, 2), _DEFAULT_HALF_WIDTH_M, dtype=np.float64)
        # Default width per side (used when CSV has no width columns)
        self._default_width: float = _DEFAULT_HALF_WIDTH_M

        # Last known positions per car (for off-track check in DNF)
        self._last_poses: np.ndarray = np.zeros((_MAX_SLOTS, 3))

        # Aggregated per-decision-step wall/vehicle flags (OR over sub-steps).
        # check_ttc_jit returns False when vel==0, so a wall hit in sub-step 1 that
        # zeros velocity would be lost by sub-step 2 without this accumulator.
        self._wall_hit:    np.ndarray = np.zeros(_MAX_SLOTS, dtype=bool)
        self._vehicle_hit: np.ndarray = np.zeros(_MAX_SLOTS, dtype=bool)

        self._step_count:  int   = 0
        self._friction:    float = _FRICTION_HI
        self._friction_countdown: int = self._next_friction_steps()

        # Deterministic LiDAR noise: generated once per simulation_step, reused by get_obs
        self._last_scan_noise: np.ndarray | None = None

        # Wall pinning flag (DESIGN.md § "Wall response")

        self._load_waypoints()
        self._jit_warmup()

    def _jit_warmup(self) -> None:
        """Compile njit helpers so SubprocVecEnv workers inherit a hot cache.

        Without this, each forked worker recompiles the njit function from scratch and races
        for the disk cache (NUMBA_CACHE_DIR). A single compile in the parent is inherited via
        copy-on-write memory and costs nothing in the children.
        """
        dummy_traj = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        _project_progress_njit(0.5, 0.0, dummy_traj,
                               np.array([0.0, 1.0]), 1.0)
        _project_lateral_offset_njit(0.5, 0.0, dummy_traj)

    # ------------------------------------------------------------------
    # Friction helpers
    # ------------------------------------------------------------------
    def _next_friction_steps(self) -> int:
        lo = int(_FRICTION_INTERVAL_SEC[0] * _DECISION_FREQ_HZ)
        hi = int(_FRICTION_INTERVAL_SEC[1] * _DECISION_FREQ_HZ)
        return int(self._rng.integers(lo, hi + 1))

    # ------------------------------------------------------------------
    # Waypoint / centerline
    # ------------------------------------------------------------------
    def _set_waypoints(self, wpts: np.ndarray) -> None:
        """Set _waypoints/_arc/_total_arc on THIS instance (single write path)."""
        # FIX A1: centerline CSVs are open polylines (last point ~one step before
        # the first).  Without the closing segment, projection in that ~0.4 m seam
        # saturates to t=0/t=1 instead of varying continuously, which freezes
        # progress at 0.0/1.0 right before the start line.  Close the loop when the
        # endpoints are within one discretisation step of each other.
        if wpts.shape[0] >= 3:
            gap = float(np.hypot(*(wpts[0] - wpts[-1])))
            first_seg = float(np.hypot(*(wpts[1] - wpts[0])))
            if 0.0 < gap <= 2.0 * max(first_seg, 1e-9):
                wpts = np.vstack([wpts, wpts[0]])
        diffs = np.diff(wpts, axis=0)
        seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
        arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
        self._waypoints = np.ascontiguousarray(wpts, dtype=np.float64)
        self._arc        = arc
        self._total_arc  = float(arc[-1])
        # Initialize default track widths if not already set
        n_segs = wpts.shape[0] - 1
        if n_segs > 0:
            self._track_widths = np.full((n_segs, 2), self._default_width, dtype=np.float64)

    def _load_waypoints(self) -> None:
        # Every track ships a comma-separated centerline CSV; the loadtxt below is
        # only a fallback for an exotic waypoint format.
        wpts = _parse_centerline_csv(_WPT_PATH)
        if wpts is not None:
            self._set_waypoints(wpts)
            self._load_track_widths(_WPT_PATH)   # colonnes 3-4 : w_right, w_left
            return
        data = np.loadtxt(
            _WPT_PATH,
            delimiter=_WPT_DELIM,
            skiprows=_WPT_ROWSKIP,
        )
        xs = data[:, _WPT_XIND]
        ys = data[:, _WPT_YIND]
        wpts = np.stack([xs, ys], axis=1)  # (N, 2)
        self._set_waypoints(wpts)

    def _compute_progress(self, x: float, y: float) -> float:
        """Project (x,y) onto centerline → progress ∈ [0,1]."""
        return float(_project_progress_njit(x, y, self._waypoints, self._arc, self._total_arc))

    def _load_track_widths(self, csv_path: str) -> None:
        """Parse track widths (columns 3-4: w_right, w_left) from centerline CSV.
        
        Falls back to _default_width if CSV has no width columns.
        """
        widths = []
        try:
            with open(csv_path) as _f:
                for line in _f:
                    stripped = line.strip()
                    if stripped.startswith("#") or not stripped:
                        continue
                    parts = [p.strip() for p in stripped.split(",")]
                    if len(parts) >= 4:
                        widths.append([float(parts[2]), float(parts[3])])
                    else:
                        break  # Not enough columns for width data
        except (IOError, ValueError):
            return  # Fallback to defaults
        
        if len(widths) > 0:
            self._track_widths = np.array(widths, dtype=np.float64)
        # else: keep defaults set by _set_waypoints

    def _is_off_track(self, x: float, y: float) -> bool:
        """Check if (x,y) is outside track boundaries using centerline projection."""
        n_segs = self._track_widths.shape[0]
        if n_segs == 0:
            return False  # no track loaded → can't be off-track
        lateral, best_seg = _project_lateral_offset_njit(x, y, self._waypoints)
        best_seg = int(best_seg)
        if best_seg >= n_segs:
            best_seg = n_segs - 1
        if best_seg < 0:
            best_seg = 0
        w_right = self._track_widths[best_seg, 0] + _OFFTRACK_MARGIN_M
        w_left = self._track_widths[best_seg, 1] + _OFFTRACK_MARGIN_M
        # Off-track if lateral offset exceeds track width (with margin) on either side
        return lateral > w_right or lateral < -w_left

    # ------------------------------------------------------------------
    # Start poses — 2×2 staggered grid (DESIGN.md § "Start grid (2×2)")
    # ------------------------------------------------------------------
    def _start_poses(self, n: int) -> np.ndarray:
        """(n, 3) start poses on a 2×2 staggered grid.

        For start index k in 0..n-1: r = k // 2 (row), c = k % 2 (column).
        Lateral offset = (c - 0.5) * _GRID_LAT_M  (perpendicular to start heading)
        Longitudinal offset = -r * _GRID_ROW_M     (behind, down the line)
        Heading = _MAP_STHETA for all cars.

        Optional robustness (nice-to-have): probe occupancy grid at candidate lateral
        poses and shrink _GRID_LAT_M if a cell is occupied — defer to avoid blocking.
        TODO: optional grid collision probe for tighter lateral placement on narrow tracks.
        """
        poses = np.zeros((n, 3))
        cos_h = math.cos(_MAP_STHETA)
        sin_h = math.sin(_MAP_STHETA)
        for k in range(n):
            r = k // 2          # row (0 or 1)
            c = k % 2           # column (0 or 1)
            lat = (c - 0.5) * _GRID_LAT_M       # left/right of racing line
            lon = -r * _GRID_ROW_M              # behind, down the line
            poses[k, 0] = _MAP_SX + lat * (-sin_h) + lon * cos_h
            poses[k, 1] = _MAP_SY + lat * cos_h  + lon * sin_h
            poses[k, 2] = _MAP_STHETA
        return poses

    # ------------------------------------------------------------------
    # Build / lidar override
    # ------------------------------------------------------------------
    def _apply_lidar_override(self) -> None:
        """
        Rebuild RaceCar class-level scanner to 100 beams.
        Must be called after EVERY Simulator (re)construction.
        Multiple Simulator instances in one interpreter share these class vars.
        """
        nb, fov = _LIDAR_RAYS, _LIDAR_SCAN_FOV
        RaceCar.scan_simulator = ScanSimulator2D(nb, fov)
        RaceCar.scan_simulator.set_map(_MAP_YAML, _MAP_EXT)
        incr = RaceCar.scan_simulator.get_increment()

        RaceCar.cosines       = np.zeros((nb,))
        RaceCar.scan_angles   = np.zeros((nb,))
        RaceCar.side_distances = np.zeros((nb,))

        dist_sides = _PARAMS["width"] / 2.0
        dist_fr    = _PARAMS["length"] / 2.0

        for i in range(nb):
            angle = -fov / 2.0 + i * incr
            RaceCar.scan_angles[i] = angle
            RaceCar.cosines[i]     = np.cos(angle)
            if angle > 0:
                if angle < math.pi / 2:
                    to_side = dist_sides / math.sin(angle)
                    to_fr   = dist_fr   / math.cos(angle)
                else:
                    to_side = dist_sides / math.cos(angle - math.pi / 2.0)
                    to_fr   = dist_fr   / math.sin(angle - math.pi / 2.0)
            else:
                if angle > -math.pi / 2:
                    to_side = dist_sides / math.sin(-angle)
                    to_fr   = dist_fr   / math.cos(-angle)
                else:
                    to_side = dist_sides / math.cos(-angle - math.pi / 2.0)
                    to_fr   = dist_fr   / math.sin(-angle - math.pi / 2.0)
            RaceCar.side_distances[i] = min(to_side, to_fr)

    def _build(self, num_agents: int) -> None:
        """Construct a fresh Simulator and apply the lidar override."""
        self._sim = Simulator(
            _PARAMS,
            num_agents=num_agents,
            seed=self._engine_seed,
            time_step=0.01,
            integrator=Integrator.RK4,
        )
        self._sim.set_map(_MAP_YAML, _MAP_EXT)
        self._apply_lidar_override()
        for agent in self._sim.agents:
            agent.ttc_thresh = _TTC_THRESH
        self._num_agents = num_agents

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, num_cars: int, seed: int | None = None) -> None:
        num_cars = max(1, min(num_cars, _MAX_SLOTS))

        # An explicit seed re-arms BOTH noise sources. Without one, the wrapper's
        # stream carries over between episodes (original behaviour).
        reseed = seed is not None
        if reseed:
            self._wrapper_seed = int(seed)
            self._engine_seed  = int(seed)
            self._rng = np.random.default_rng(seed=self._wrapper_seed)

        if self._sim is None or num_cars != self._num_agents or reseed:
            self._build(num_cars)

        poses = self._start_poses(num_cars)
        self._sim.reset(poses)  # type: ignore[union-attr]
        for i in range(num_cars):
            self._sim.agents[i].ttc_thresh = _TTC_THRESH

        # Reset per-agent bookkeeping
        for i in range(_MAX_SLOTS):
            self._actions[i]             = np.zeros(2)
            self._steering_cmd[i]        = 0.0
            self._progress[i]            = 0.0          # rel phase (own-start)
            self._rel_prev[i]            = 0.0          # previous rel (own-start)
            self._cum[i]                 = 0.0          # cumulative fractional laps from own start
            self._last_delta[i]          = 0.0          # signed wrap-aware per-step delta
            self._lap_count[i]           = 0
            self._last_lap_complete[i]   = False
            self._status[i]              = 1 if i < num_cars else 0
            self._lap_times[i].clear()
            self._lap_start_step[i] = 0
            self._finish_step[i]    = -1
            self._stagnation_flag[i] = False
            self._max_progress[i]      = 0.0          # progress-based DNF reset

        self._step_count = 0
        self._friction   = _FRICTION_HI
        self._friction_countdown = self._next_friction_steps()

        # Apply initial friction immediately (not waiting for first countdown interval)
        p_init = dict(_PARAMS)
        p_init["mu"] = self._friction
        self._sim.update_params(p_init, agent_idx=-1)  # type: ignore[union-attr]

        # Record absolute progress_start per car and compute initial rel phase
        for i in range(num_cars):
            x = float(self._sim.agents[i].state[0])
            y = float(self._sim.agents[i].state[1])
            p0 = self._compute_progress(x, y)
            self._progress_start[i] = p0
            self._rel_prev[i]       = 0.0          # (p0 - p0) mod 1 == 0
            self._cum[i]            = 0.0
            self._lap_count[i]      = 0
            self._progress[i]       = 0.0          # rel at start is 0
            self._last_delta[i]     = 0.0
            self._progress_history[i].clear()
            self._max_progress[i]   = 0.0          # progress-based DNF reset per car

    # ------------------------------------------------------------------
    # apply_action
    # ------------------------------------------------------------------
    def apply_action(self, agent_id: int, target_speed: float, steering: float) -> None:
        if self._status[agent_id] != 1:   # only ACTIVE cars accept actions
            return
        # Wrapper bounds (see get_space_info), NOT engine v_min/v_max.
        speed = float(np.clip(target_speed, _TARGET_SPEED_MIN, _TARGET_SPEED_MAX))
        steer = float(np.clip(steering, _PARAMS["s_min"], _PARAMS["s_max"]))
        self._actions[agent_id, 0]  = steer   # engine: steer first
        self._actions[agent_id, 1]  = speed
        self._steering_cmd[agent_id] = steer

    # ------------------------------------------------------------------
    # DNF helpers
    # ------------------------------------------------------------------
    def _dnf_pose(self, slot: int) -> tuple[float, float, float]:
        """Off-map pose used for frozen DNF'd agents.

        Positioning far from any track feature means the car never participates in
        GJK/EPAD overlap checks or LiDAR scans. It's purely to keep the physics world
        clean and avoid accidental collision reports with active cars.
        """
        return (10000.0 + slot * 5.0, 10000.0, 0.0)

    def _freeze_dnf(self, i: int) -> None:
        """Deactivate slot i: move off-map, zero velocity, mark DNF."""
        self._status[i] = 0
        if i < self._num_agents:
            x, y, _ = self._dnf_pose(i)
            agent = self._sim.agents[i]  # type: ignore[union-attr]
            agent.state[0] = x
            agent.state[1] = y
            agent.state[3:] = 0.0

    def _finish(self, i: int) -> None:
        """Required laps reached: status=2 (FINISHED), removed from scene, frozen."""
        self._status[i] = 2
        self._finish_step[i] = self._step_count
        if i < self._num_agents:
            x, y, _ = self._dnf_pose(i)
            a = self._sim.agents[i]
            a.state[0] = x; a.state[1] = y; a.state[3:] = 0.0

    # ------------------------------------------------------------------
    # Knockback helper (vehicle-vehicle impulse exchange)
    # ------------------------------------------------------------------

    def _is_inside_wall(self, x: float, y: float) -> bool:
        """True if (x, y) falls in an occupied map cell.

        Reads the scanner's distance transform (0 m inside obstacles): one array
        lookup, no ray casting. Off-map is not "in a wall" — xy_2_rc returns
        (-1, -1) there, and dt[-1, -1] would be unrelated.
        """
        sim = RaceCar.scan_simulator
        if sim is None or sim.dt is None:
            return False
        r, c = xy_2_rc(
            x, y,
            sim.orig_x, sim.orig_y, sim.orig_c, sim.orig_s,
            sim.map_height, sim.map_width, sim.map_resolution,
        )
        if r < 0 or c < 0:
            return False
        return bool(sim.dt[r, c] <= 0.0)

    def _apply_knockback(self, i: int, j: int) -> None:
        """
        Position-based knockback with subtle velocity modulation.

        Main effect: positional displacement along the contact normal.
        Subtle effect: velocity magnitude modulated ±15% of impulse, always preserving
        the original sign of the forward speed to avoid unwanted orientation changes.
        """
        si = self._sim.agents[i].state  # type: ignore[union-attr]
        sj = self._sim.agents[j].state  # type: ignore[union-attr]

        pi = np.array([si[0], si[1]])
        pj = np.array([sj[0], sj[1]])
        d  = pi - pj
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            return
        n = d / dist  # contact normal: j→i

        # relative velocity along normal
        vi_world = si[3] * np.array([math.cos(si[4]), math.sin(si[4])])
        vj_world = sj[3] * np.array([math.cos(sj[4]), math.sin(sj[4])])
        vrel = float(np.dot(vi_world - vj_world, n))
        if vrel >= 0.0:
            return  # already separating
        vrel = max(abs(vrel), 0.1)

        impulse_mag = min(abs(vrel) * _KNOCKBACK_REST, _KNOCKBACK_MAX_M)

        # Split weighted by momentum actually aimed at the other car, not by total
        # speed: a fast car going past in parallel does not shove. Unchanged for
        # head-on and rear-end contacts, where velocities already align with n.
        speed_i = abs(float(np.dot(vi_world, n)))
        speed_j = abs(float(np.dot(vj_world, n)))
        speed_sum = speed_i + speed_j
        # Neither closing on the other -> split evenly.
        share_i = (speed_j / speed_sum) if speed_sum > 1e-5 else 0.5
        # Clamped so neither car walks away untouched: ramming a stopped car gave
        # its share a near-zero weight, so the rammer paid nothing.
        share_i = min(max(share_i, _KNOCKBACK_MIN_SHARE), 1.0 - _KNOCKBACK_MIN_SHARE)
        impulse_mag_j_to_i = impulse_mag * share_i
        impulse_mag_i_to_j = impulse_mag * (1.0 - share_i)
        
        # --- Position displacement (main effect) ---
        si[0] += impulse_mag_j_to_i * n[0]
        si[1] += impulse_mag_j_to_i * n[1]
        sj[0] -= impulse_mag_i_to_j * n[0]
        sj[1] -= impulse_mag_i_to_j * n[1]

        # --- Subtle velocity influence ---
        # Each car is nudged along its own push direction (i along +n, j along -n),
        # projected onto its heading since state[3] is speed along the chassis axis.
        # Testing both against +n sped the rammer UP in a rear-end.
        vel_change_i = impulse_mag_j_to_i * 0.15
        vel_change_j = impulse_mag_i_to_j * 0.15
        head_i = np.array([math.cos(si[4]), math.sin(si[4])])
        head_j = np.array([math.cos(sj[4]), math.sin(sj[4])])

        v_max = _PARAMS["v_max"]
        si[3] = float(np.clip(si[3] + vel_change_i * float(np.dot(n, head_i)), -v_max, v_max))
        sj[3] = float(np.clip(sj[3] - vel_change_j * float(np.dot(n, head_j)), -v_max, v_max))

    # ------------------------------------------------------------------
    # simulation_step
    # ------------------------------------------------------------------
    def simulation_step(self) -> None:
        """
        Advance physics by one decision step (5 sub-steps at 100 Hz = 10 ms each = 50 ms
        total, action held constant). On wall contact (agent.in_collision), restores the
        full pose (x, y, yaw) from a pre-sub-step snapshot to prevent the engine's yaw-
        clobber-on-collision from causing heading corruption and off-track escape.
        """
        assert self._sim is not None, "Call reset() before simulation_step()"

        n = self._num_agents

        # Build control matrix; DNF'd slots get zero action
        ctrl = np.zeros((n, 2))
        for i in range(n):
            if self._status[i] == 1:
                ctrl[i] = self._actions[i]
            # DNF'd agents keep ctrl=[0,0]; engine still steps them (frozen off-map)

        # Reset per-decision-step collision accumulators before the sub-step loop.
        # check_ttc_jit returns False when vel==0, so a wall hit that zeros velocity in
        # sub-step k would be invisible by sub-step k+1 without this OR accumulation.
        self._wall_hit[:] = False
        self._vehicle_hit[:] = False

        # Snapshots per active agent: (x, y, yaw) before each sub-step — used to un-pinch
        # wall tunneling where check_ttc clobbers state[4]=yaw to 0, snapping heading to +x.
        # The RaceCar state layout is [x, y, steer, vel, yaw, yaw_rate, slip] (idx3=vel).
        # check_ttc in_collision → state[3:]=0 zeroes vel AND yaw. Once yaw=0 the car points
        # into open space and escapes. Fix: restore x, y AND yaw on wall contact.
        snap_x = np.zeros(_MAX_SLOTS)
        snap_y = np.zeros(_MAX_SLOTS)
        snap_yaw = np.zeros(_MAX_SLOTS)
        snap_vel = np.zeros(_MAX_SLOTS)

        # Pairs (i, j) already knocked back during this decision step.
        knocked_pairs: set[tuple[int, int]] = set()

        for sub in range(_PHYSICS_STEPS):
            # Snapshot active agents BEFORE the step
            for i in range(n):
                if self._status[i] == 1:
                    snap_x[i] = self._sim.agents[i].state[0]
                    snap_y[i] = self._sim.agents[i].state[1]
                    snap_yaw[i] = self._sim.agents[i].state[4]
                    snap_vel[i] = self._sim.agents[i].state[3]

            obs = self._sim.step(ctrl)
            # OR collision flags across sub-steps
            for i in range(n):
                if self._status[i] == 1:
                    self._wall_hit[i] |= bool(self._sim.agents[i].in_collision)
                    self._vehicle_hit[i] |= bool(int(self._sim.collision_idx[i]) >= 0)
            # Per-agent collision response after each sub-step (active agents only)
            for i in range(n):
                if self._status[i] != 1:
                    continue
                wall = bool(self._sim.agents[i].in_collision)

                if wall:
                    # FIX C5: exact rollback to the pre-sub-step legal pose (DESIGN
                    # § Wall response). No backward offset, no pinned flag — pinning
                    # is emergent (re-collision each sub-step until the car reverses).
                    # Penetration cannot accumulate: a pose ending in collision is
                    # never validated as the next sub-step's starting point.
                    self._sim.agents[i].state[0] = snap_x[i]
                    self._sim.agents[i].state[1] = snap_y[i]
                    self._sim.agents[i].state[4] = snap_yaw[i]
                    # Wall bounce: engine zeroed vel; give back part of it,
                    # reversed, so the car can back out (vel==0 => no yaw either).
                    self._sim.agents[i].state[3] = -min(
                        _WALL_REST * abs(snap_vel[i]), _WALL_REST_MAX
                    )
                    # yaw_rate/slip stay zeroed by engine check_ttc
                    ctrl[i, 1] = 0.0

            # --- Vehicle-vehicle knockback, AT the sub-step of contact ---
            # collision_idx is recomputed every sub-step and reset to -1 once contact
            # ends, so reading it here (not after the loop) always yields a valid
            # partner and catches brief rubs. knocked_pairs caps the effect at one
            # knockback per pair per decision step.
            for i in range(n):
                if self._status[i] != 1:
                    continue
                j = int(self._sim.collision_idx[i])
                if j < 0 or j >= n or i >= j or self._status[j] != 1:
                    continue
                if (i, j) in knocked_pairs:
                    continue
                knocked_pairs.add((i, j))
                self._apply_knockback(i, j)
                # Knockback moves the pose with no occupancy check, so a car may end
                # up in a wall (intended). Flag it now, else it surfaces a step late.
                for k in (i, j):
                    s = self._sim.agents[k].state
                    if self._is_inside_wall(float(s[0]), float(s[1])):
                        self._wall_hit[k] = True

            # Re-freeze DNF/FINISHED slots every sub-step (separate pass, all agents)
            for j in range(n):
                if self._status[j] != 1:
                    x_dnf, y_dnf, _ = self._dnf_pose(j)
                    self._sim.agents[j].state[0] = x_dnf
                    self._sim.agents[j].state[1] = y_dnf
                    self._sim.agents[j].state[3:] = 0.0

        # --- Post-step: sync obs poses for active agents (reflects any pose rollback) ---
        # _update_progress_and_laps reads obs["poses_x/y/theta"] — ensure they match
        # the (possibly restored) state so progress deltas are correct.
        for i in range(n):
            if self._status[i] == 1:
                obs["poses_x"][i] = self._sim.agents[i].state[0]
                obs["poses_y"][i] = self._sim.agents[i].state[1]
                obs["poses_theta"][i] = self._sim.agents[i].state[4]

        self._step_count += 1

        # --- Bookkeeping at decision rate ---
        # Changes: self._friction, self._friction_countdown
        self._update_friction()
        # Changes: self._progress, self._rel_prev, self._cum, self._last_delta, self._lap_count, 
        #          self._last_lap_complete, self._finish_step, self._max_progress (reset on lap complete)
        self._update_progress_and_laps(obs)
        # Changes: self._stagnation_flag, self._status (can set to 0/DNF), self._progress_history, 
        #          self._max_progress
        self._update_stagnation_and_dnf()

        # Generate deterministic LiDAR noise once per step (reused by get_obs)
        # Changes: self._last_scan_noise
        self._update_lidar_noise()

    # ------------------------------------------------------------------
    # Weather (ground friction + LiDAR noise)
    # ------------------------------------------------------------------
    def _update_friction(self) -> None:
        # TODO: friction is pinned to _FRICTION_HI for the whole run. Make mu vary
        # during the race and apply it with self._sim.update_params(p, agent_idx=-1).
        p = dict(_PARAMS)
        p["mu"] = _FRICTION_HI

    def _update_lidar_noise(self) -> None:
        # TODO: the noise half-range is pinned to _LIDAR_NOISE for the whole run.
        # Make it vary during the race.
        self._last_scan_noise = self._rng.uniform(-_LIDAR_NOISE, _LIDAR_NOISE, size=_LIDAR_RAYS)


    # ------------------------------------------------------------------
    # Progress & lap detection — per-car odometer (DESIGN.md § "Lap detection")
    # ------------------------------------------------------------------
    def _update_progress_and_laps(self, obs: dict) -> None:
        n = self._num_agents
        for i in range(n):
            if self._status[i] != 1:
                self._last_delta[i] = 0.0
                self._last_lap_complete[i] = False   # FIX A5: unlatch, else it stays sticky
                continue

            x = float(obs["poses_x"][i])
            y = float(obs["poses_y"][i])
            self._last_poses[i, 0] = x
            self._last_poses[i, 1] = y
            p_abs = self._compute_progress(x, y)              # absolute, [0,1)
            rel   = (p_abs - self._progress_start[i]) % 1.0   # own-start phase, [0,1)
            # FIX A1: for a tiny positive progress_start, (0.0 - p0) % 1.0 rounds to
            # exactly 1.0 — which would latch _max_progress at its ceiling and force
            # a stagnation DNF one window later.  Keep the phase half-open.
            if rel >= 1.0:
                rel = 0.0

            d     = rel - self._rel_prev[i]
            if d < -0.5:
                d += 1.0          # forward wrap
            elif d > 0.5:
                d -= 1.0         # backward wrap

            self._cum[i]       += d
            self._last_delta[i] = d   # store for get_step_info progress_delta
            self._rel_prev[i]   = rel
            self._progress[i]   = rel   # obs["progress"] = rel

            new_count = int(np.floor(self._cum[i] + 1e-9))
            if new_count > self._lap_count[i]:                # latched, non-decreasing
                # one lap (or more) booked this step
                self._last_lap_complete[i] = True
                # FIX E5: no reset here — the odometer is continuous across the
                # line, so there is no discontinuity left to compensate (was C1).
                lap_time = (self._step_count - self._lap_start_step[i]) / _DECISION_FREQ_HZ
                self._lap_times[i].append(float(lap_time))
                self._lap_start_step[i] = self._step_count
                self._lap_count[i] = new_count
                # FIX A2: test the integer counter, not the raw float. `_cum >= 3`
                # disagreed with the tolerant floor(cum + 1e-9) above, so a crossing
                # at cum=2.9999999999999956 booked the lap but missed the finish.
                if new_count >= _REQUIRED_LAPS:
                    self._finish(i)
            else:
                self._last_lap_complete[i] = False

    # ------------------------------------------------------------------
    # Stagnation + DNF detection (combined)
    # ------------------------------------------------------------------
    def _update_stagnation_and_dnf(self) -> None:
        n = self._num_agents
        for i in range(n):
            if self._status[i] != 1:
                self._stagnation_flag[i] = False
                continue

            x = float(self._last_poses[i, 0])
            y = float(self._last_poses[i, 1])

            # Off-track DNF: immediate when outside track boundaries
            if self._is_off_track(x, y):
                self._freeze_dnf(i)
                continue

            # FIX E5: running max on the odometer, not on _progress — that phase
            # wraps to ~0.999 when reversing past one's own start, latching the max
            # at its ceiling and forcing a DNF one window later.
            self._max_progress[i] = max(float(self._cum[i]), self._max_progress[i])

            # Record running max in history deque (maxlen=_DNF_WINDOW_STEPS+1)
            self._progress_history[i].append(self._max_progress[i])

            if len(self._progress_history[i]) < _DNF_WINDOW_STEPS + 1:
                self._stagnation_flag[i] = False
                continue

            # DNF: strictly increasing max-progress required over window.
            # Require a minimum gain, not just any increase: a car pinned against a
            # wall creeps by a millimetre per step and never trips a strict compare.
            window_start = self._progress_history[i][0]
            min_gain = _DNF_MIN_GAIN_M / max(self._total_arc, 1e-9)
            stag = (self._max_progress[i] - window_start) < min_gain
            self._stagnation_flag[i] = stag
            if stag:
                self._freeze_dnf(i)

    # ------------------------------------------------------------------
    # Rankings — by cumulative distance (DESIGN.md § "Ranking by cum")
    # ------------------------------------------------------------------
    def _compute_ranks(self) -> dict:
        n = self._num_agents
        finished = [i for i in range(n) if self._status[i] == 2]
        active   = [i for i in range(n) if self._status[i] == 1]
        dnf      = [i for i in range(n) if self._status[i] == 0]
        finished.sort(key=lambda i: self._finish_step[i])              # earlier finish = better
        active.sort(key=lambda i: (-self._cum[i], i))                  # highest cum = best, tie-break slot
        # FIX E6: DNFs were left in slot order. Rank them on distance too, but on
        # _max_progress, not _cum: a car shoved backwards by the crash that killed
        # it keeps credit for how far it actually got. Same key as sim_recorder.
        dnf.sort(key=lambda i: (-self._max_progress[i], i))
        ordered = finished + active + dnf
        ranks = {}
        for rank, i in enumerate(ordered, start=1):
            ranks[i] = rank
        for i in range(_MAX_SLOTS):
            if i not in ranks:
                ranks[i] = _MAX_SLOTS
        return ranks

    # ------------------------------------------------------------------
    # Stagnation flag reader (returns stored flag, not recomputed)
    # ------------------------------------------------------------------
    def _stagnation(self) -> dict:
        return {i: bool(self._stagnation_flag[i]) for i in range(_MAX_SLOTS)}

    # ------------------------------------------------------------------
    # get_obs
    # ------------------------------------------------------------------
    def get_obs(self, agent_id: int) -> dict:
        # See agent.state in base_classes.py
        assert self._sim is not None, "Call reset() before get_obs()"

        n = self._num_agents
        agent = self._sim.agents[agent_id]

        # LiDAR scan (from last step observation via agent_scans stored in agent)
        # Re-scan at current pose using the class-var scanner + rng
        scan_pose = np.array([
            float(agent.state[0]) + 0.0 * math.cos(float(agent.state[4])),
            float(agent.state[1]) + 0.0 * math.sin(float(agent.state[4])),
            float(agent.state[4]),
        ])
        raw_scan = RaceCar.scan_simulator.scan(scan_pose, None)  # no engine noise here
        # Reuse deterministic noise generated in simulation_step() (None before first step → zero noise)
        if self._last_scan_noise is not None:
            noise = 1.0 + self._last_scan_noise
        else:
            noise = np.ones(_LIDAR_RAYS)
        scan     = np.clip(raw_scan * noise, _LIDAR_MIN, _LIDAR_MAX).astype(np.float32)

        velocity = float(agent.state[3])
        steering = float(agent.state[2])
        progress = float(self._progress[agent_id])   # own-start rel phase
        lap_count = int(self._lap_count[agent_id])

        ranks = self._compute_ranks()
        rank  = ranks.get(agent_id, _MAX_SLOTS)

        # Opponents (ego body frame, fixed keys 0..3)
        ego_x    = float(agent.state[0])
        ego_y    = float(agent.state[1])
        ego_yaw  = float(agent.state[4])
        cos_e    = math.cos(ego_yaw)
        sin_e    = math.sin(ego_yaw)

        opponents: dict = {}
        for slot in range(_MAX_SLOTS):
            if slot == agent_id or slot >= n or self._status[slot] != 1:
                opponents[slot] = {
                    "x_rel": 0.0, "y_rel": 0.0, "yaw_rel": 0.0,
                    "speed": 0.0, "progress": 0.0, "lap_count": 0,
                    "status": (0 if (slot == agent_id or slot >= n) else int(self._status[slot])),
                }
                continue
            opp  = self._sim.agents[slot]
            dx   = float(opp.state[0]) - ego_x
            dy   = float(opp.state[1]) - ego_y
            x_rel =  cos_e * dx + sin_e * dy
            y_rel = -sin_e * dx + cos_e * dy
            diff_yaw  = float(opp.state[4]) - ego_yaw
            yaw_rel   = math.atan2(math.sin(diff_yaw), math.cos(diff_yaw))
            speed_opp = float(opp.state[3])
            opponents[slot] = {
                "x_rel":     x_rel,
                "y_rel":     y_rel,
                "yaw_rel":   yaw_rel,
                "speed":     speed_opp,
                "progress":  float(self._progress[slot]),   # opponent's own-start phase
                "lap_count": int(self._lap_count[slot]),
                "status":    int(self._status[slot]),
            }

        return {
            "agent_id":  agent_id,
            "lidar":     scan,
            "velocity":  velocity,
            "steering":  steering,
            "progress":  progress,
            "lap_count": lap_count,
            "rank":      rank,
            "opponents": opponents,
        }

    # ------------------------------------------------------------------
    # get_step_info
    # ------------------------------------------------------------------
    def get_step_info(self) -> dict:
        assert self._sim is not None, "Call reset() before get_step_info()"

        n = self._num_agents
        ranks = self._compute_ranks()
        stagnation = self._stagnation()

        wall_col: dict    = {}
        vehicle_col: dict = {}
        prog_delta: dict  = {}
        lap_complete: dict = {}

        for i in range(_MAX_SLOTS):
            if i < n:
                # wall = "hit a wall while moving", not "touching a wall": the engine's
                # iTTC test is velocity-dependent, so a car stopped against a wall
                # reports False. Sparse by design (see DESIGN.md).
                wall_col[i]    = bool(self._wall_hit[i])
                vehicle_col[i] = bool(self._vehicle_hit[i])
                # Use the stored per-step signed wrap-aware delta from odometer
                prog_delta[i] = float(self._last_delta[i])
                # FIX A5: no status guard — _finish() flips status to 2 on the same
                # step the last lap is booked, so the guard ate the winning lap.
                lap_complete[i] = bool(self._last_lap_complete[i])
            else:
                wall_col[i]    = False
                vehicle_col[i] = False
                prog_delta[i]  = 0.0
                lap_complete[i] = False

        opponents_mask = np.zeros(_MAX_SLOTS, dtype=bool)
        for i in range(_MAX_SLOTS):
            opponents_mask[i] = bool(i < n and self._status[i] == 1)

        agent_status: dict = {}
        for i in range(_MAX_SLOTS):
            agent_status[i] = int(self._status[i]) if i < n else 0

        return {
            "step_count":       self._step_count,
            "time_elapsed":     float(self._step_count / _DECISION_FREQ_HZ),
            "ranks":            ranks,
            "collisions": {
                "wall":    wall_col,
                "vehicle": vehicle_col,
            },
            "progress_delta":   prog_delta,
            "lap_complete":     lap_complete,
            "stagnation":       stagnation,
            "friction_current": self._friction,
            "opponents_mask":   opponents_mask,
            "agent_status":     agent_status,
            "lap_times": {i: list(self._lap_times[i]) for i in range(_MAX_SLOTS)},
            "race_over": bool(all(self._status[i] != 1 for i in range(n))),
            "max_progress":     {i: float(self._max_progress[i]) for i in range(_MAX_SLOTS)},
        }

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._sim = None
        self._num_agents = 0


# ---------------------------------------------------------------------------
# Module-level singleton and 6 public functions
# ---------------------------------------------------------------------------
_sim_instance: _Sim | None = None


def _get_sim() -> _Sim:
    global _sim_instance
    if _sim_instance is None:
        _sim_instance = _Sim()
    return _sim_instance


def get_space_info() -> dict:
    """Return observation / action bounds and metadata. Call once at startup."""
    return {
        "observations": {
            "lidar":     {"shape": (_LIDAR_RAYS,), "bounds": (0.0, _LIDAR_MAX), "unit": "meters"},
            "velocity":  {"shape": "scalar", "bounds": (_VEL_OBS_MIN, _VEL_OBS_MAX),   "unit": "m/s"},
            "steering":  {"shape": "scalar", "bounds": (_PARAMS["s_min"], _PARAMS["s_max"]), "unit": "rad"},
            "progress":  {"shape": "scalar", "bounds": (0.0, 1.0),    "unit": "normalized lap"},
            "lap_count": {"shape": "scalar", "bounds": (0, math.inf), "unit": "int"},
            "opponents": {"shape": "dict[0..3]",                      "unit": "dict of {x_rel, y_rel, yaw_rel, speed, progress, lap_count, status}"},
        },
        "actions": {
            "target_speed": {"bounds": (_TARGET_SPEED_MIN, _TARGET_SPEED_MAX), "unit": "m/s target velocity (asymmetric)"},
            "steering":     {"bounds": (_PARAMS["s_min"], _PARAMS["s_max"]), "unit": "rad"},
        },
        "decision_freq_hz": _DECISION_FREQ_HZ,
    }


def reset(num_cars: int = 1, seed: int | None = None) -> None:
    """
    Reset simulation. Lazily rebuilds Simulator if num_cars changed.
    num_cars ∈ [1, 4]. Friction resets to 1.0. All per-agent state cleared.

    seed : RNG seed (LiDAR noise). Given, it re-arms both seeds and makes the
    episode exactly replayable. Omitted, the noise stream carries over.
    """
    _get_sim().reset(num_cars, seed)


def get_obs(agent_id: int) -> dict:
    """
    Full state observation for one agent.
    LiDAR is wall-only (multiplicative noise of half-range _LIDAR_NOISE, clipped to [0.1, 15.0] m).
    Opponents are in ego body frame (fixed keys 0..3).
    """
    return _get_sim().get_obs(agent_id)


def apply_action(agent_id: int, target_speed: float, steering: float) -> None:
    """
    Register (target_speed, steering) for agent_id.
    Silently clamped: speed ∈ [-2.0, 10.0] m/s, steering ∈ [-0.4189, 0.4189] rad.
    Engine input format is [steer, speed]; assembly happens inside simulation_step().
    Call before simulation_step(), once per agent per loop.
    """
    _get_sim().apply_action(agent_id, target_speed, steering)


def simulation_step() -> None:
    """
    Advance physics by one decision step (5 × 10 ms sub-steps = 50 ms).
    Action held constant across sub-steps. Call exactly once per loop.
    """
    _get_sim().simulation_step()


def get_step_info() -> dict:
    """
    Global per-step information: ranks, collisions (wall/vehicle separated),
    progress_delta, lap_complete, stagnation, friction, masks, status.
    No 'done' field — termination is the student's responsibility.
    """
    return _get_sim().get_step_info()


def close() -> None:
    """Destroy the simulation and free engine state. Call at script end."""
    global _sim_instance
    if _sim_instance is not None:
        _sim_instance.close()
        _sim_instance = None


# ---------------------------------------------------------------------------
# Map switching (DESIGN.md § "Switching Map Circuits")
#   set_map() loads a map + triggers Simulator rebuild. Idempotent — no-op
#   if the requested map is already loaded. reset() reuses whatever is active.
# ---------------------------------------------------------------------------

def _load_config_override(path: str) -> tuple[float, float, float] | None:
    """Try to read sx/sy/stheta from a config YAML. Return None on failure."""
    try:
        import yaml as _ym
        with open(path) as _f:
            _c = _ym.safe_load(_f)
        return (float(_c["sx"]), float(_c["sy"]), float(_c["stheta"]))
    except Exception:
        return None


def set_map(name: str, num_cars: int | None = None) -> None:
    """Load a map circuit. Call once before training starts. Idempotent (no-op if same map).

    Args:
        name: Map directory name, e.g. ``"Spa"``, ``"Catalunya"``, ``"example"``.
              Use :func:`get_available_maps` to list available tracks.

    Raises:
        ValueError: If the map name is unknown (not in maps/ and not ``"example"``).

    Performance:
        Triggers a single Simulator rebuild (~5–20 ms, paid once per map switch).
        Repeated calls with the same name are no-op — tracked by ``_current_map``.

    Per-map structure:
        maps/<name>/<name>_centerline.csv  — centerline (optional; origin fallback if missing)
        maps/<name>/<name>_map.png         — occupancy grid image
        maps/<name>/<name>_map.yaml        — map yaml (resolution/origin/offsets)
        maps/<name>/<name>_config.yaml     — start-pose overrides (optional; sx/sy/stheta)
    """
    global _current_map, _MAP_YAML, _MAP_EXT, _WPT_PATH, _MAP_CONFIG_YAML
    global _MAP_SX, _MAP_SY, _MAP_STHETA

    if name == _current_map:
        return  # already loaded — no-op

    available = _ensure_maps_discovered()
    if name not in available and name != "example":
        raise ValueError(f"Unknown map '{name}'. Available: {get_available_maps()}")

    # FIX E1: set_map is atomic. Any failure restores the previous state and
    # re-raises, instead of letting _current_map advertise an unloaded track.
    _prev = (_current_map, _MAP_YAML, _MAP_EXT, _WPT_PATH, _MAP_CONFIG_YAML,
             _MAP_SX, _MAP_SY, _MAP_STHETA)
    _prev_map_name  = _sim_instance._map_name   if _sim_instance is not None else None
    _prev_num_cars  = _sim_instance._num_agents if _sim_instance is not None else 0
    _prev_had_sim   = _sim_instance is not None and _sim_instance._sim is not None

    try:
        if name == "example":
            # Default / fallback map (from maps/examples/)
            base_dir = _EXAMPLES_DIR
            config_base = "example_map"
        else:
            base_dir = os.path.join(_REPO_ROOT, "maps", name)
            config_base = name + "_map"

        _MAP_YAML   = os.path.join(base_dir, f"{config_base}.yaml")
        _MAP_EXT    = ".png"
        _WPT_PATH   = os.path.join(base_dir, f"{config_base.replace('_map', '_centerline')}.csv")
        _MAP_CONFIG_YAML = os.path.join(base_dir, f"{config_base.replace('_map', '_config')}.yaml")

        # FIX E1: fail now. Without this check a track with missing files is
        # accepted while no Simulator exists (nothing reads disk before reset()),
        # and the next call, believing it loaded, does nothing.
        for _needed in (_MAP_YAML, os.path.join(base_dir, f"{config_base}{_MAP_EXT}")):
            if not os.path.exists(_needed):
                raise FileNotFoundError(
                    f"Map '{name}' is incomplete: missing {_needed}")

        # Try loading sx/sy/stheta from the config YAML; fall back to map data.
        cfg = _load_config_override(_MAP_CONFIG_YAML)
        if cfg is not None:
            _MAP_SX, _MAP_SY, _MAP_STHETA = cfg
        elif name != "example":
            # Derive start pose from the map's own data (centerline CSV or map YAML origin).
            centerline_path = os.path.join(base_dir, f"{name}_centerline.csv")
            map_yaml_path   = os.path.join(base_dir, f"{name}_map.yaml")

            if os.path.exists(centerline_path):
                # Use first non-comment waypoint as start point.
                with open(centerline_path) as _f:
                    for line in _f:
                        stripped = line.strip()
                        if stripped.startswith("#") or not stripped:
                            continue
                        parts = [p.strip() for p in stripped.split(",")]
                        if len(parts) >= 2:
                            _MAP_SX = float(parts[0])
                            _MAP_SY = float(parts[1])
                            # Compute heading from the first two waypoints.
                            with open(centerline_path) as _f2:
                                wpts_raw = []
                                for l in _f2:
                                    s = l.strip()
                                    if s.startswith("#") or not s:
                                        continue
                                    pp = [p.strip() for p in s.split(",")]
                                    if len(pp) >= 2:
                                        wpts_raw.append((float(pp[0]), float(pp[1])))
                            if len(wpts_raw) >= 2:
                                dx = wpts_raw[1][0] - wpts_raw[0][0]
                                dy = wpts_raw[1][1] - wpts_raw[0][1]
                                _MAP_STHETA = float(math.atan2(dy, dx))
                            else:
                                _MAP_STHETA = _DEFAULT_THETA
                            break
                    else:
                        # No waypoints found in CSV → fall through to map_yaml origin.
                        _load_start_from_map_yaml(base_dir, name)
            else:
                _load_start_from_map_yaml(base_dir, name)
        else:
            _MAP_SX, _MAP_SY, _MAP_STHETA = 0.7, 0.0, 1.37079632679

        # Re-warmup the njit progress projector with the new centerline (if it exists).
        if name != "example" and os.path.exists(_WPT_PATH):
            _rebuild_waypoints_from_csv(_WPT_PATH)

        _current_map = name
        # FIX C3b/C3c: rebuild the Simulator on the new map. Invalidating _sim forces
        # reset() → _build(), which re-reads the updated _MAP_YAML globals for both the
        # Simulator occupancy grid and the ScanSimulator2D (LiDAR + iTTC). num_cars is
        # honored (lazy rebuild with the requested count).
        if _sim_instance is not None:
            _sim_instance._map_name = name
            if name == "example":
                _sim_instance._load_waypoints()   # restore example centerline
            n = num_cars if num_cars is not None else max(_sim_instance._num_agents, 1)
            _sim_instance._sim = None
            _sim_instance.reset(n)
    except Exception:
        (_current_map, _MAP_YAML, _MAP_EXT, _WPT_PATH, _MAP_CONFIG_YAML,
         _MAP_SX, _MAP_SY, _MAP_STHETA) = _prev
        if _sim_instance is not None:
            _sim_instance._map_name = _prev_map_name
            # _build() may have left a half-built Simulator: rebuild it on the
            # previous track, else _sim stays None (a clean error on the next step
            # rather than an inconsistent grid).
            _sim_instance._sim = None
            if _prev_had_sim and _current_map is not None:
                # Same calls as the nominal path, to restore the exact state.
                if _current_map == "example":
                    _sim_instance._load_waypoints()
                elif os.path.exists(_WPT_PATH):
                    _rebuild_waypoints_from_csv(_WPT_PATH)
                try:
                    _sim_instance.reset(max(_prev_num_cars, 1))
                except Exception:
                    _sim_instance._sim = None
        raise


# ---------------------------------------------------------------------------
# Helpers for deriving start pose when no config YAML exists
# ---------------------------------------------------------------------------
_DEFAULT_THETA = 1.37079632679  # ~78.5 degrees (example map default heading)

def _load_start_from_map_yaml(base_dir: str, name: str) -> None:
    """Derive start pose from the map YAML origin when no config/centerline is available."""
    global _MAP_SX, _MAP_SY, _MAP_STHETA   # FIX C3d: assignments were silently local
    map_yaml_path = os.path.join(base_dir, f"{name}_map.yaml")
    if os.path.exists(map_yaml_path):
        import yaml as _ym
        with open(map_yaml_path) as _f:
            mdata = _ym.safe_load(_f)
        if mdata and isinstance(mdata, dict) and "origin" in mdata:
            orig = mdata["origin"]
            # Use map origin + small lateral offset as start point.
            _MAP_SX = float(orig[0])
            _MAP_SY = float(orig[1])
            _MAP_STHETA = _DEFAULT_THETA
    else:
        _MAP_SX, _MAP_SY, _MAP_STHETA = 0.7, 0.0, _DEFAULT_THETA


def _parse_centerline_csv(csv_path: str) -> np.ndarray | None:
    """Parse a comma-delimited centerline CSV ('#' comments) → (N,2) array or None."""
    rows = []
    with open(csv_path) as _f:
        for line in _f:
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) >= 2:
                rows.append([float(parts[0]), float(parts[1])])
    if len(rows) < 2:
        return None  # Not enough waypoints
    return np.array(rows, dtype=np.float64)


def _rebuild_waypoints_from_csv(csv_path: str) -> None:
    """Rebuild the singleton's _waypoints/_arc/_total_arc from a centerline CSV.

    FIX C3a: writes on the _Sim INSTANCE via _set_waypoints. The previous version
    wrote on the _Sim CLASS attributes, which the instance attributes (set in
    __init__) shadowed — progress kept being projected on the old centerline.
    """
    wpts = _parse_centerline_csv(csv_path)
    if wpts is None:
        return
    if _sim_instance is not None:
        _sim_instance._set_waypoints(wpts)
        _sim_instance._load_track_widths(csv_path)   # _set_waypoints reset it to the default
    # If the singleton doesn't exist yet, _Sim.__init__ → _load_waypoints will
    # parse the same CSV itself (map-aware) — nothing to do here.


def get_available_maps() -> list[str]:
    """Return a sorted list of available circuit names for :func:`set_map`.

    Discovers maps under ``maps/<Name>/`` (one-shot, cached), plus the built-in
    ``"example"`` map. ``__pycache__`` is excluded.
    """
    return sorted(_ensure_maps_discovered() | {"example"})


def get_start_pose(slot_index: int) -> tuple[float, float, float]:
    """Return (x, y, yaw) for 2×2 grid slot index (0..3).

    Works at any time (does NOT require reset() or a running simulator).
    The start pose is the same one that reset() uses internally.

    Grid layout:
        Slot 0 ── Slot 1       (front row, lateral ±_GRID_LAT_M/2 from center)
        Slot 2 ── Slot 3       (back row, _GRID_ROW_M behind front row)

    All cars share the same heading (_MAP_STHETA = track tangent at start point).
    """
    if not (0 <= slot_index <= 3):
        raise ValueError(f"slot_index must be 0..3, got {slot_index}")

    r = slot_index // 2          # row (0 or 1)
    c = slot_index % 2           # column (0 or 1)
    lat = (c - 0.5) * _GRID_LAT_M    # perpendicular to start heading
    lon = -r * _GRID_ROW_M            # behind, along -heading

    cos_h = math.cos(_MAP_STHETA)
    sin_h = math.sin(_MAP_STHETA)

    x = _MAP_SX + lat * (-sin_h) + lon * cos_h
    y = _MAP_SY + lat * cos_h  + lon * sin_h
    return (x, y, _MAP_STHETA)


def get_current_map() -> str | None:
    """Return the name of the currently loaded map, or None if no map has been set.

    This is a module-level convenience function that does NOT require a running simulator.
    Use it to verify which map circuit is active before starting training or a tournament.
    """
    return _current_map
