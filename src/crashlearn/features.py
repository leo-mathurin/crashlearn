"""Canonical observation features (E-16). Layout and rationale: docs/features.md.

Pure NumPy and the standard library on purpose: the export (E-25) copies this module into
every submission `agent.py`, where only NumPy and ONNX Runtime are allowed. It must not
import the simulator; tests check that the constants below still match it.

One fixed-size input for every driver, solo or champion: the opponent slots are always there,
masked when empty, so a solo policy transfers to the champion without changing its
architecture (ADR 0001).

Tolerant reading: a missing key or a non-finite value becomes the neutral value instead of
raising, because an exception in the tournament means a DNF.
"""

import math

import numpy as np

LIDAR_RAYS = 100
LIDAR_MAX = 15.0  # m
SPEED_MAX = 10.0  # m/s
STEER_MAX = 0.4189  # rad
REQUIRED_LAPS = 3
MAX_OPPONENTS = 3
STATUS_ACTIVE = 1

EGO_SIZE = LIDAR_RAYS + 5
# mask, x_rel, y_rel, sin yaw_rel, cos yaw_rel, speed, race gap
OPPONENT_SIZE = 7
FEATURE_SIZE = EGO_SIZE + MAX_OPPONENTS * OPPONENT_SIZE


def _number(mapping, key: str, default: float = 0.0) -> float:
    try:
        value = float(mapping.get(key, default))
    except (AttributeError, TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _lidar(obs: dict) -> np.ndarray:
    """100 ranges in metres. Missing rays read as free space, the reading of an empty scan."""
    try:
        rays = np.asarray(obs.get("lidar", ()), dtype=np.float64).ravel()
    except (TypeError, ValueError):
        rays = np.empty(0)
    out = np.full(LIDAR_RAYS, LIDAR_MAX)
    out[: min(rays.size, LIDAR_RAYS)] = rays[:LIDAR_RAYS]
    return np.nan_to_num(out, nan=LIDAR_MAX, posinf=LIDAR_MAX, neginf=0.0)


def adapt_observation(obs: dict) -> dict:
    """Raw `get_obs` dict -> canonical form with explicit units and a fixed opponent table.

    `opponents` is a (MAX_OPPONENTS, 6) array (x_rel, y_rel, yaw_rel, speed, progress,
    lap_count) of the active opponents, nearest first, zero-padded; `mask` marks the filled
    rows. The ego slot and inactive cars (status != 1) are left out, whatever their values.
    """
    raw = obs.get("opponents")
    slots = raw.values() if isinstance(raw, dict) else ()
    active = []
    for slot in slots:
        if not isinstance(slot, dict) or _number(slot, "status") != STATUS_ACTIVE:
            continue
        keys = ("x_rel", "y_rel", "yaw_rel", "speed", "progress", "lap_count")
        active.append([_number(slot, k) for k in keys])
    # nearest first, ties broken on the values themselves: the slot order never matters
    active.sort(key=lambda row: (math.hypot(row[0], row[1]), row))
    active = active[:MAX_OPPONENTS]

    opponents = np.zeros((MAX_OPPONENTS, 6))
    mask = np.zeros(MAX_OPPONENTS, dtype=bool)
    if active:
        opponents[: len(active)] = active
        mask[: len(active)] = True
    return {
        "lidar": _lidar(obs),
        "velocity": _number(obs, "velocity"),
        "steering": _number(obs, "steering"),
        "progress": _number(obs, "progress"),
        "lap_count": _number(obs, "lap_count"),
        "opponents": opponents,
        "mask": mask,
    }


def build_features(obs: dict) -> np.ndarray:
    """Raw `get_obs` dict of one car -> float32 vector of FEATURE_SIZE values in [-1, 1]."""
    c = adapt_observation(obs)
    phase = 2.0 * math.pi * c["progress"]
    ego_laps = c["lap_count"] + c["progress"]
    ego = [
        c["velocity"] / SPEED_MAX,
        c["steering"] / STEER_MAX,
        math.sin(phase),
        math.cos(phase),
        c["lap_count"] / REQUIRED_LAPS,
    ]
    rows = np.zeros((MAX_OPPONENTS, OPPONENT_SIZE))
    for i in np.flatnonzero(c["mask"]):
        x, y, yaw, speed, progress, laps = c["opponents"][i]
        rows[i] = [
            1.0,
            x / LIDAR_MAX,
            y / LIDAR_MAX,
            math.sin(yaw),
            math.cos(yaw),
            speed / SPEED_MAX,
            # laps ahead (+) or behind (-), from each car's own start: a few metres off on
            # a staggered grid, enough to tell ahead from lapped
            laps + progress - ego_laps,
        ]
    vec = np.concatenate([c["lidar"] / LIDAR_MAX, ego, rows.ravel()])
    return np.clip(vec, -1.0, 1.0).astype(np.float32)
