"""Deterministic LiDAR gap-follow driver: a diagnostic baseline, never a submission.

Used for control races (is a metric or the simulator wrong, or the RL policy?) and as a simple
league opponent. The subject requires learned policies: this driver never replaces them.

Same call signature as the tournament agents: `predict(obs, info) -> (target_speed, steering)`.
NumPy and the standard library only; all state lives on the instance.
Every action goes through one clamp, so the contract holds for any DriverConfig variant.
Observations must have the simulator's shape: `opponents` as a dict, an integer `agent_id` and
a numeric LiDAR. Other shapes (list, None, text) are not supported; harden only if the league
ever takes observations from external sources.
"""

import math
from dataclasses import dataclass

import numpy as np

LIDAR_RAYS = 100
LIDAR_MIN, LIDAR_MAX = 0.1, 15.0
SPEED_MIN, SPEED_MAX = -2.0, 10.0
STEER_MAX = 0.4189
# Ray i points at -pi + i * 2pi/99 in the car frame (0 = ahead, positive = left, like steering).
ANGLES = -math.pi + np.arange(LIDAR_RAYS) * (2 * math.pi / (LIDAR_RAYS - 1))
ANGLE_STEP = 2 * math.pi / (LIDAR_RAYS - 1)
CAR_RADIUS = 0.35  # opponent footprint (car is 0.58 x 0.31 m)


@dataclass(frozen=True)
class DriverConfig:
    """Every tuning value. Bump `version` whenever a default changes (league results cite it)."""

    version: str = "1"
    fov_deg: float = 100.0  # half-angle of the window where a heading target is chosen
    disparity_threshold: float = 0.5  # m, range jump treated as an obstacle edge
    car_half_width: float = 0.16  # m
    margin: float = 0.25  # m, extra clearance added when extending disparities
    target_tolerance: float = 0.5  # m, rays this close to the deepest one are candidates
    steer_gain: float = 1.0  # steering = gain * target angle
    front_cone_deg: float = 10.0  # half-angle used to measure free distance ahead
    front_margin: float = 1.0  # m, free distance at which speed is min_speed
    speed_gain: float = 1.5  # m/s of target speed per metre of free distance
    min_speed: float = 1.0  # m/s
    max_speed: float = 7.0  # m/s
    steer_slowdown: float = 0.5  # speed factor lost at full lock
    stuck_speed: float = 0.2  # m/s, below this the car counts as stuck
    stuck_steps: int = 10  # decisions stuck before reversing
    # Below |v| = 0.5 m/s the engine uses its kinematic model. Faster steered reversing makes
    # the dynamic single-track model diverge (yaw rate -> 1e30, see docs): stay under it.
    reverse_speed: float = -0.4  # m/s
    reverse_steps: int = 20  # decisions spent reversing
    # A gap follower has no sense of direction: after a contact it may turn around. Progress
    # falling for this many decisions while moving forward triggers a full-lock U-turn.
    wrong_way_steps: int = 5
    # Reversing right after the start/finish line makes the simulator book ~1.0 max progress
    # and DNF the car 4 s later (see docs/scripted_driver.md): never reverse below this.
    no_reverse_below_progress: float = 0.02


class ScriptedDriver:
    def __init__(self, config: DriverConfig | None = None) -> None:
        self.config = config or DriverConfig()
        self.reset()

    def reset(self) -> None:
        self._stuck = 0
        self._reverse_left = 0
        self._reverse_steer = 0.0
        self._backward = 0
        self._prev_progress = 0.0

    def predict(self, obs: dict, info: dict) -> tuple[float, float]:
        speed, steer = self._decide(obs, info)
        # single exit clamp: both driving directions, any config, non-finite -> 0
        return (
            float(np.clip(_finite(speed, 0.0), SPEED_MIN, SPEED_MAX)),
            float(np.clip(_finite(steer, 0.0), -STEER_MAX, STEER_MAX)),
        )

    def _decide(self, obs: dict, info: dict) -> tuple[float, float]:
        c = self.config
        if info.get("step_count") == 0:  # a missing key is not a new episode
            self.reset()
        me = int(obs.get("agent_id", 0))
        scan = self._scan_with_opponents(obs, me)
        steer = self._steering(scan)
        velocity = _finite(obs.get("velocity", 0.0), 0.0)
        progress = _finite(obs.get("progress", 0.0), 0.0)
        delta = (progress - self._prev_progress + 0.5) % 1.0 - 0.5  # lap wrap-aware
        self._prev_progress = progress
        if velocity > c.stuck_speed and delta < 0:
            self._backward += 1
        elif delta > 0:
            self._backward = 0
        if self._backward >= c.wrong_way_steps:
            rear = np.flatnonzero(np.abs(ANGLES) > math.pi / 2)
            steer = math.copysign(STEER_MAX, ANGLES[rear[np.argmax(scan[rear])]])

        if self._reverse_left > 0:
            self._reverse_left -= 1
            return c.reverse_speed, self._reverse_steer

        self._stuck = self._stuck + 1 if abs(velocity) < c.stuck_speed else 0
        if _wall_hit(info, me) or self._stuck >= c.stuck_steps:
            self._stuck = 0
            if progress >= c.no_reverse_below_progress:
                # reversing with opposite lock points the nose back toward the free direction
                self._reverse_left = c.reverse_steps - 1
                self._reverse_steer = -steer
                return c.reverse_speed, self._reverse_steer
            return c.min_speed, steer
        if self._backward >= c.wrong_way_steps:
            return c.min_speed, steer

        front = scan[_cone(c.front_cone_deg)].min()
        speed = min(c.min_speed + c.speed_gain * (front - c.front_margin), c.max_speed)
        speed *= 1.0 - c.steer_slowdown * abs(steer) / STEER_MAX
        return max(speed, c.min_speed), steer

    def _scan_with_opponents(self, obs: dict, me: int) -> np.ndarray:
        """Sanitised scan (NaN/inf -> bounds) plus active opponents: the LiDAR sees walls only."""
        raw = np.asarray(obs.get("lidar", ()), dtype=np.float64).reshape(-1)
        scan = np.full(LIDAR_RAYS, LIDAR_MIN)
        n = min(raw.size, LIDAR_RAYS)
        scan[:n] = np.nan_to_num(raw[:n], nan=LIDAR_MIN, posinf=LIDAR_MAX, neginf=LIDAR_MIN)
        scan = np.clip(scan, LIDAR_MIN, LIDAR_MAX)
        for slot, opp in (obs.get("opponents") or {}).items():
            if slot == me or opp.get("status", 0) != 1:
                continue
            x, y = _finite(opp.get("x_rel", 0.0), 0.0), _finite(opp.get("y_rel", 0.0), 0.0)
            dist = math.hypot(x, y)
            if dist < 1e-6:
                continue
            half = math.atan2(CAR_RADIUS, dist)
            diff = np.abs((ANGLES - math.atan2(y, x) + math.pi) % (2 * math.pi) - math.pi)
            hit = diff <= half
            scan[hit] = np.minimum(scan[hit], max(dist - CAR_RADIUS, LIDAR_MIN))
        return scan

    def _steering(self, scan: np.ndarray) -> float:
        """Disparity extender over the front window, then aim at the deepest ray nearest ahead."""
        c = self.config
        window = np.flatnonzero(_cone(c.fov_deg))
        d = scan[window]
        ext = d.copy()
        for k in range(len(d) - 1):
            near, far = (k, k + 1) if d[k] < d[k + 1] else (k + 1, k)
            if d[far] - d[near] <= c.disparity_threshold:
                continue
            rays = math.ceil(math.atan2(c.car_half_width + c.margin, d[near]) / ANGLE_STEP)
            step = 1 if far > near else -1
            end = min(max(near + step * rays, 0), len(d) - 1)
            span = slice(min(far, end), max(far, end) + 1)
            ext[span] = np.minimum(ext[span], d[near])
        candidates = np.flatnonzero(ext >= ext.max() - c.target_tolerance)
        angles = ANGLES[window][candidates]
        target = angles[np.argmin(np.abs(angles))]  # ties: lowest index (right) wins, stable
        return float(np.clip(c.steer_gain * target, -STEER_MAX, STEER_MAX))


def _cone(half_angle_deg: float) -> np.ndarray:
    """Rays within the half-angle; never empty (the two rays around straight ahead stay in)."""
    return np.abs(ANGLES) <= max(math.radians(half_angle_deg), ANGLE_STEP)


def _finite(value, default: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _wall_hit(info: dict, me: int) -> bool:
    try:
        return bool(info["collisions"]["wall"][me])
    except (KeyError, IndexError, TypeError):
        return False
