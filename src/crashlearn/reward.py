"""Race reward, versioned (E-15). Formula, scales and rationale: docs/mdp.md.

Every term is computed and returned separately so that training can log it. The total is
their plain sum: no term is hidden in another one.

What the simulator credits is what we reward (ADR 0004): signed progress along the
centerline, booked laps, the finish. Contacts are logged with a zero weight until E-47 shows
that they cost more than they pay.
"""

import dataclasses
import math

import numpy as np

STATUS_DNF, STATUS_ACTIVE, STATUS_FINISHED = 0, 1, 2
DECISION_DT = 0.05  # s, 20 Hz decisions


@dataclasses.dataclass(frozen=True)
class RewardConfig:
    version: str = "v1"
    # +1 per metre of signed centerline progress: the same metre is worth the same on every
    # track, and reversing over a stretch takes back exactly what driving it gave
    progress_per_m: float = 1.0
    # one-off events, latched by the simulator: a lap cannot be booked twice
    lap_bonus: float = 20.0
    finish_bonus: float = 50.0
    dnf_penalty: float = -200.0
    # logged, not weighted (ADR 0004, E-47)
    wall_contact: float = 0.0
    vehicle_contact: float = 0.0
    # light per-step shaping, small next to the ~0.3 m gained per step at 6 m/s
    slow_below_mps: float = 1.0
    slow_penalty: float = -0.1
    reverse_below_mps: float = -0.1  # an idle car hovers around -1e-3 m/s
    reverse_grace_steps: int = 20  # 1 s of reversing out of a wall is free
    reverse_penalty: float = -0.1
    action_change_penalty: float = -0.02  # x squared change of the normalized action


TERMS = (
    "progress",
    "lap",
    "finish",
    "dnf",
    "wall_contact",
    "vehicle_contact",
    "slow",
    "reverse",
    "action_change",
)


class RaceReward:
    """Stateful per-episode reward of one car: call `reset` at every episode start."""

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()
        self.reset(track_length_m=1.0)

    def reset(self, track_length_m: float) -> None:
        if not track_length_m > 0:
            raise ValueError(f"track length must be positive, got {track_length_m}")
        self.track_length_m = float(track_length_m)
        self._reverse_steps = 0
        self._prev_action: np.ndarray | None = None

    def __call__(
        self, obs: dict, info: dict, ego_id: int, action: np.ndarray
    ) -> tuple[float, dict[str, float]]:
        """(raw obs, raw step info, ego id, normalized action) -> (total, terms)."""
        c = self.config
        status = info["agent_status"][ego_id]
        progress_m = info["progress_delta"][ego_id] * self.track_length_m

        velocity = obs["velocity"]
        self._reverse_steps = self._reverse_steps + 1 if velocity < c.reverse_below_mps else 0
        action = np.asarray(action, dtype=np.float64)
        change = (
            0.0 if self._prev_action is None else float(np.sum((action - self._prev_action) ** 2))
        )
        self._prev_action = action

        running = status == STATUS_ACTIVE
        terms = {
            "progress": c.progress_per_m * progress_m,
            "lap": c.lap_bonus * bool(info["lap_complete"][ego_id]),
            "finish": c.finish_bonus * (status == STATUS_FINISHED),
            "dnf": c.dnf_penalty * (status == STATUS_DNF),
            "wall_contact": c.wall_contact * bool(info["collisions"]["wall"][ego_id]),
            "vehicle_contact": c.vehicle_contact * bool(info["collisions"]["vehicle"][ego_id]),
            "slow": c.slow_penalty * (running and progress_m < c.slow_below_mps * DECISION_DT),
            "reverse": c.reverse_penalty * (self._reverse_steps > c.reverse_grace_steps),
            "action_change": c.action_change_penalty * change,
        }
        terms = {k: float(v) for k, v in terms.items()}
        total = math.fsum(terms.values())
        return total, terms
