"""Per-car temporal memory (E-13). Reset rules and their limits: docs/history.md.

Pure NumPy and the standard library on purpose, like `features.py`: the export (E-25) copies
this module into every submission `agent.py` that drives with a history. The tournament may
call one `Agent` instance for several cars (`sim_recorder` replicates a flat submission
across every car), so the memory is keyed by `obs["agent_id"]`, never kept on the instance.

A new episode starts the car's history over: its first frame is repeated `DEPTH` times, so
the stack has a fixed size from the first decision. The reactive driver (A) keeps no state
and stays the fallback when a reset cannot be detected.

Tolerant reading: missing or invalid keys never raise, because an exception in the
tournament means a DNF.
"""

import math

import numpy as np

DEPTH = 4
DECISION_FREQ_HZ = 20
# Key of the cars whose obs has no usable agent_id: kept apart from a real car 0.
MISSING_ID = -1
# agent_loader.create_dummy_obs, sent once to predict() before the race: every ray at 5 m.
PROBE_LIDAR = 5.0


def _get(mapping, key: str):
    try:
        return mapping.get(key)
    except AttributeError:
        return None


def _integer(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) and number == int(number) else None


def car_id(obs: dict) -> int:
    """`obs["agent_id"]`, or MISSING_ID when it is absent or not a non-negative integer."""
    value = _integer(_get(obs, "agent_id"))
    return value if value is not None and value >= 0 else MISSING_ID


def decision_step(info: dict) -> int | None:
    """Step counter of `get_step_info`: `step_count`, else `time_elapsed` x 20 Hz, else None."""
    step = _integer(_get(info, "step_count"))
    if step is not None:
        return step
    try:
        elapsed = float(_get(info, "time_elapsed"))
    except (TypeError, ValueError):
        return None
    return round(elapsed * DECISION_FREQ_HZ) if math.isfinite(elapsed) else None


def is_probe(obs: dict, info: dict) -> bool:
    """True for the loader's dry-run call, whose dummy frame must not enter any history."""
    if decision_step(info) not in (0, None) or _get(obs, "velocity") != 0.0:
        return False
    try:
        rays = np.asarray(_get(obs, "lidar"), dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return rays.size > 0 and bool(np.all(rays == PROBE_LIDAR))


class CarHistory:
    """The last DEPTH feature vectors of each car, oldest first."""

    def __init__(self, depth: int = DEPTH):
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.depth = depth
        self._frames: dict[int, np.ndarray] = {}
        self._last_step: dict[int, int | None] = {}
        self._last_laps: dict[int, int] = {}

    def reset(self) -> None:
        self._frames.clear()
        self._last_step.clear()
        self._last_laps.clear()

    def push(self, obs: dict, info: dict, features: np.ndarray) -> np.ndarray:
        """Add this decision's features to the car of `obs`; return its float32 stack.

        The result has `depth * features.size` values: the oldest frame first, `features`
        last. A dry-run probe gets a stack of its own frame and leaves the memory untouched.
        """
        frame = np.asarray(features, dtype=np.float32).ravel()
        if is_probe(obs, info):
            return np.tile(frame, self.depth)

        car = car_id(obs)
        step = decision_step(info)
        laps = _integer(_get(obs, "lap_count"))
        frames = self._frames.get(car)
        if frames is None or frames.shape[1] != frame.size or self._new_episode(car, step, laps):
            frames = np.tile(frame, (self.depth, 1))
        else:
            frames = np.concatenate([frames[1:], frame[None]])

        self._frames[car] = frames
        self._last_step[car] = step
        if laps is not None:
            self._last_laps[car] = laps
        return frames.ravel().copy()

    def _new_episode(self, car: int, step: int | None, laps: int | None) -> bool:
        last = self._last_step.get(car)
        # Each car decides once per step: anything but the next step is another episode,
        # or a gap after which the memory no longer describes the present.
        if step is not None and last is not None and step != last + 1:
            return True
        last_laps = self._last_laps.get(car)
        return laps is not None and last_laps is not None and laps < last_laps
