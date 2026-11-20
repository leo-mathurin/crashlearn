"""Weather that changes during the race (E-23): schedules and their effect on the engine."""

import env_simulation as es
import numpy as np
import pytest

from crashlearn.env import CrashLearnEnv, controls_to_action
from crashlearn.weather import (
    DECISION_FREQ_HZ,
    TOURNAMENT_FRICTION,
    TOURNAMENT_LIDAR_NOISE,
    Segment,
    WeatherConfig,
    schedule,
)

STEPS = 8000


def draw(seed: int, config: WeatherConfig | None = None) -> list[Segment]:
    return schedule(config or WeatherConfig(), np.random.default_rng(seed), STEPS)


def test_a_schedule_covers_the_race_inside_the_tournament_ranges():
    for seed in range(50):
        segments = draw(seed)
        assert segments[0].start_step == 0
        starts = [s.start_step for s in segments]
        assert starts == sorted(set(starts)) and starts[-1] < STEPS
        gaps = np.diff([*starts, STEPS])
        assert np.all(gaps[:-1] >= 5 * DECISION_FREQ_HZ)
        assert np.all(gaps[:-1] <= 40 * DECISION_FREQ_HZ)
        for s in segments:
            assert TOURNAMENT_FRICTION[0] <= s.friction <= TOURNAMENT_FRICTION[1]
            assert TOURNAMENT_LIDAR_NOISE[0] <= s.lidar_noise <= TOURNAMENT_LIDAR_NOISE[1]


def test_friction_and_noise_change_together_or_separately():
    kinds = set()
    for seed in range(20):
        segments = draw(seed)
        for before, after in zip(segments, segments[1:], strict=False):
            kinds.add((before.friction != after.friction, before.lidar_noise != after.lidar_noise))
    assert kinds == {(True, False), (False, True), (True, True)}


def test_the_same_seed_gives_the_same_weather():
    assert draw(3) == draw(3) != draw(4)


def test_a_nominal_episode_keeps_the_official_values():
    segments = draw(0, WeatherConfig(nominal_share=1.0))
    assert segments == [Segment(0, 1.0, 0.001)]


@pytest.mark.parametrize(
    "kwargs", [{"friction": (0.0, 1.0)}, {"segment_s": (5.0, 1.0)}, {"nominal_share": 1.5}]
)
def test_invalid_configs_are_refused(kwargs):
    with pytest.raises(ValueError):
        WeatherConfig(**kwargs)


@pytest.fixture
def make_env():
    envs = []

    def build(**kwargs):
        env = CrashLearnEnv(maps="Sochi", max_episode_steps=600, **kwargs)
        envs.append(env)
        return env

    yield build
    for env in envs:
        env.close()


FIXED = [Segment(0, 1.0, 0.001), Segment(100, 0.7, 0.1), Segment(200, 0.85, 0.001)]


def test_a_segment_reaches_the_engine_and_the_observation_at_its_first_decision(make_env):
    env = make_env()
    _, info = env.reset(seed=0, options={"weather": FIXED})
    action = controls_to_action(1.0, 0.0)
    seen = [(info["friction"], info["lidar_noise"])]
    for _ in range(250):
        _, _, _, _, info = env.step(action)
        mu = es._get_sim()._sim.agents[0].params["mu"]
        assert info["step_info"]["friction_current"] == info["friction"] == mu
        seen.append((info["friction"], info["lidar_noise"]))
    # seen[k] holds after k decisions: the segment of step 100 drives decision 100
    assert seen[100] == (1.0, 0.001) and seen[101] == (0.7, 0.1)
    assert seen[201] == (0.85, 0.001) and seen[-1] == (0.85, 0.001)


def test_changing_friction_changes_the_dynamics(make_env):
    def pose(weather):
        env = make_env()
        env.reset(seed=0, options={"weather": weather})
        # 25 decisions: still moving, before the wall stops the car
        for _ in range(25):
            env.step(controls_to_action(4.0, 0.4189))
        result = es._get_sim()._sim.agents[0].state[[0, 1, 4]].copy()
        env.close()
        return result

    dry = [Segment(0, 1.0, 0.001)]
    rain_late = [Segment(0, 1.0, 0.001), Segment(10, 0.7, 0.001)]
    assert not np.allclose(pose(dry), pose(rain_late), atol=1e-3)


def test_training_weather_is_replayable_from_the_episode_seed(make_env):
    env = make_env(weather=WeatherConfig(segment_s=(1.0, 2.0)))
    runs = []
    for _ in range(2):
        _, info = env.reset(seed=5)
        values = [(info["friction"], info["lidar_noise"])]
        for _ in range(200):
            info = env.step(controls_to_action(1.0, 0.0))[4]
            values.append((info["friction"], info["lidar_noise"], *info["raw_obs"]["lidar"][:3]))
        runs.append(values)
    assert runs[0] == runs[1]
    assert len({v[:2] for v in runs[0]}) > 2


def test_weather_replaces_the_fixed_overrides():
    with pytest.raises(ValueError, match="weather replaces"):
        CrashLearnEnv(maps="Sochi", friction=0.8, weather=WeatherConfig())
