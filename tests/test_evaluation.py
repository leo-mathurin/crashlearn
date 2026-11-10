"""Evaluation: identical races for every driver, failures kept."""

import json

import env_simulation as es
import numpy as np
import pytest

from crashlearn import tracks
from crashlearn.env import CrashLearnEnv, controls_to_action
from crashlearn.evaluation import (
    CONDITIONS,
    EVAL_SEEDS,
    STRICT_OFFTRACK_MARGIN_M,
    Race,
    ScriptedPilot,
    campaign,
    evaluate,
    run_race,
)
from crashlearn.history_env import HistoryWrapper


class Fixed:
    def __init__(self, speed: float, steering: float = 0.0):
        self.action = controls_to_action(speed, steering)

    def reset(self) -> None:
        pass

    def act(self, obs, info):
        return self.action


@pytest.fixture(scope="module", autouse=True)
def _restore_shared_sim_state():
    """Closing an env destroys the singleton: give later modules the state of `conftest.sim`."""
    yield
    es.close()
    es.set_map("example")
    es.reset(num_cars=2)
    es.simulation_step()


@pytest.fixture
def env():
    environment = CrashLearnEnv(max_episode_steps=400)
    yield environment
    environment.close()


RACE = Race("train", "IMS", "nominal", 0)


def test_campaign_covers_the_protocol_and_never_the_sealed_test():
    races = campaign()
    train, validation = len(tracks.train_tracks()), len(tracks.validation_tracks())
    perturbed = len(CONDITIONS) - 1
    assert len(races) == ((train + validation) + validation * perturbed) * len(EVAL_SEEDS)
    assert {r.map for r in races} == set(tracks.train_tracks()) | set(tracks.validation_tracks())
    assert not {r.map for r in races} & set(tracks.sealed_test_tracks(unseal=True))


def test_perturbed_conditions_only_run_on_validation_tracks():
    assert {r.split for r in campaign() if r.condition != "nominal"} == {"validation"}


def test_campaign_filters_keep_the_protocol_order():
    races = campaign(maps=["Sochi"], conditions=["heavy_rain"], seeds=(1,))
    assert races == [Race("validation", "Sochi", "heavy_rain", 1)]
    assert campaign(maps=["Spa"]) == []


def test_a_timeout_is_a_record_not_an_error(env):
    result = run_race(env, Fixed(1.5), RACE)
    assert result["status"] == "TIMEOUT"
    assert result["decisions"] == 400
    assert result["total_time_s"] is None
    assert result["progress_laps"] > 0


def test_a_dnf_is_a_record_not_an_error(env):
    result = run_race(env, Fixed(0.0), RACE)
    assert result["status"] == "DNF"
    assert result["dnf_step"] == result["decisions"]
    assert result["total_time_s"] is None


@pytest.mark.parametrize("condition", list(CONDITIONS))
def test_conditions_reach_the_simulator(env, condition):
    result = run_race(env, Fixed(1.0), Race("validation", "Sochi", condition, 0))
    assert result["friction"] == CONDITIONS[condition]["friction"]
    assert result["lidar_noise"] == CONDITIONS[condition]["lidar_noise"]


def test_the_strict_margin_reaches_the_simulator_and_is_restored(env):
    default = es._OFFTRACK_MARGIN_M
    strict = run_race(env, Fixed(1.0), Race("validation", "Sochi", "strict_margin", 0))
    assert strict["offtrack_margin_m"] == STRICT_OFFTRACK_MARGIN_M < default
    assert es._OFFTRACK_MARGIN_M == default
    assert run_race(env, Fixed(1.0), RACE)["offtrack_margin_m"] == default


def test_the_strict_margin_moves_the_offtrack_line_during_the_race(env):
    """A point 0.75 m past the edge is off track at 0.50 m, still on track at 1.00 m."""
    seen = []

    class Probe(Fixed):
        def act(self, obs, info):
            s = es._get_sim()
            (x0, y0), (x1, y1) = s._waypoints[0, :2], s._waypoints[1, :2]
            nx, ny = np.array([y0 - y1, x1 - x0]) / np.hypot(x1 - x0, y1 - y0)
            for side, width in ((1, s._track_widths[0, 1]), (-1, s._track_widths[0, 0])):
                d = side * (width + 0.75)
                seen.append(s._is_off_track(x0 + d * nx, y0 + d * ny))
            return self.action

    run_race(env, Probe(0.0), Race("validation", "Sochi", "nominal", 0))
    assert seen and not any(seen)
    seen.clear()
    run_race(env, Probe(0.0), Race("validation", "Sochi", "strict_margin", 0))
    assert seen and all(seen)


def test_the_strict_margin_is_restored_when_a_race_raises(env):
    class Crash(Fixed):
        def act(self, obs, info):
            raise RuntimeError("driver crash")

    default = es._OFFTRACK_MARGIN_M
    with pytest.raises(RuntimeError):
        run_race(env, Crash(1.0), Race("validation", "Sochi", "strict_margin", 0))
    assert es._OFFTRACK_MARGIN_M == default


def test_behaviour_metrics_of_a_reversing_and_an_idle_car(env):
    backwards = run_race(env, Fixed(-2.0), RACE)
    assert backwards["reverse_steps"] > 0 and backwards["action_change_per_step"] == 0
    idle = run_race(env, Fixed(0.0), RACE)
    assert idle["idle_steps"] == idle["decisions"] and idle["reverse_steps"] == 0
    assert idle["progress_jump_steps"] == 0


def test_the_same_race_is_replayed_exactly(env):
    assert run_race(env, ScriptedPilot(), RACE) == run_race(env, ScriptedPilot(), RACE)


def test_the_seed_changes_the_race(env):
    first = run_race(env, ScriptedPilot(), RACE)
    second = run_race(env, ScriptedPilot(), Race("train", "IMS", "nominal", 1))
    assert first["trajectory_sha256"] != second["trajectory_sha256"]


def test_the_scripted_control_finishes_three_laps_with_its_documented_time():
    env = CrashLearnEnv()
    try:
        result = run_race(env, ScriptedPilot(), RACE)
    finally:
        env.close()
    assert result["status"] == "FINISHED"
    assert result["laps_completed"] == 3
    assert len(result["lap_times_s"]) == 3
    assert result["total_time_s"] == pytest.approx(132.3, abs=2.0)  # docs/scripted_driver.md


def test_every_driver_gets_the_same_start():
    env = HistoryWrapper(CrashLearnEnv(max_episode_steps=5), depth=1)
    try:
        starts = []
        for _ in range(2):
            obs, _ = env.reset(seed=4, options={"map": "IMS"})
            starts.append(obs)
        assert np.array_equal(*starts)
    finally:
        env.close()


def test_results_are_on_disk_after_each_race(env, tmp_path):
    class Crashes(Fixed):
        calls = 0

        def reset(self):
            Crashes.calls += 1
            if Crashes.calls == 2:
                raise RuntimeError("mid-campaign crash")

    output = tmp_path / "nested" / "races.json"
    races = [RACE, Race("train", "IMS", "nominal", 1)]
    with pytest.raises(RuntimeError):
        evaluate(env, Crashes(1.5), races, output, {"driver": "x"})
    kept = json.loads(output.read_text())
    assert [r["seed"] for r in kept] == [0]
    assert kept[0]["driver"] == "x"
