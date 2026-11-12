"""Weather that changes during the race (E-23, docs/weather.md).

The tournament draws ground friction in [0.7, 1.0] and a LiDAR noise half-range in
[0.001, 0.1], and changes them during the race, together or separately, at unknown moments
and for unknown durations (subject v1.3.0, p. 3 and 9). The official simulator pins both
(`_update_friction` never applies mu, `TODO` upstream), so `CrashLearnEnv` applies them from
our side, `vendor/` untouched.

A schedule is a list of constant segments drawn once per episode from one seeded generator:
the same seed gives the same weather, so a race stays replayable.
"""

from dataclasses import asdict, dataclass

import numpy as np

DECISION_FREQ_HZ = 20
# Subject v1.3.0 ranges: friction 0.7 = heavy rain, LiDAR noise 0.001 = dry.
TOURNAMENT_FRICTION = (0.7, 1.0)
TOURNAMENT_LIDAR_NOISE = (0.001, 0.1)
# What changes at each boundary: friction only, noise only, or both.
CHANGES = ("friction", "lidar_noise", "both")


@dataclass(frozen=True)
class WeatherConfig:
    friction: tuple[float, float] = TOURNAMENT_FRICTION
    lidar_noise: tuple[float, float] = TOURNAMENT_LIDAR_NOISE
    # duration of a constant segment, in seconds of race time
    segment_s: tuple[float, float] = (5.0, 40.0)
    # share of episodes kept in the official conditions (friction 1.0, noise 0.001)
    nominal_share: float = 0.0

    def __post_init__(self):
        for name in ("friction", "lidar_noise", "segment_s"):
            low, high = getattr(self, name)
            if not 0 < low <= high:
                raise ValueError(f"{name} must be a range 0 < low <= high, got {(low, high)}")
        if not 0 <= self.nominal_share <= 1:
            raise ValueError("nominal_share must be in [0, 1]")

    def to_dict(self) -> dict:
        """JSON form, lists for ranges: what a manifest records and compares on resume."""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


@dataclass(frozen=True)
class Segment:
    start_step: int
    friction: float
    lidar_noise: float


def schedule(config: WeatherConfig, rng: np.random.Generator, steps: int) -> list[Segment]:
    """Constant segments that cover `steps` decisions. The first one starts at step 0.

    `None` values never appear: a nominal episode is a single segment at friction 1.0 and
    noise 0.001, the values of the official simulator.
    """
    if rng.uniform() < config.nominal_share:
        return [Segment(0, TOURNAMENT_FRICTION[1], TOURNAMENT_LIDAR_NOISE[0])]
    friction = float(rng.uniform(*config.friction))
    noise = float(rng.uniform(*config.lidar_noise))
    segments = [Segment(0, friction, noise)]
    step = 0
    while True:
        step += max(1, round(rng.uniform(*config.segment_s) * DECISION_FREQ_HZ))
        if step >= steps:
            return segments
        change = CHANGES[int(rng.integers(len(CHANGES)))]
        if change != "lidar_noise":
            friction = float(rng.uniform(*config.friction))
        if change != "friction":
            noise = float(rng.uniform(*config.lidar_noise))
        segments.append(Segment(step, friction, noise))
