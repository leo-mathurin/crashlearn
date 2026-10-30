"""E-18 evaluation: identical races, kept failures, and the comparison rule fixed in advance."""

import json

import env_simulation as es
import numpy as np
import pytest

from crashlearn import tracks
from crashlearn.env import CrashLearnEnv, controls_to_action
from crashlearn.evaluation import (
    CONDITIONS,
    EVAL_SEEDS,
    Race,
    ScriptedPilot,
    bootstrap_interval,
    campaign,
    driver_summary,
    evaluate,
    paired_differences,
    provenance,
    run_race,
    verdict,
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


def record(
    driver, train_seed, map_name, status="FINISHED", time=100.0, seed=0, condition="nominal"
):
    return {
        "driver": driver,
        "train_seed": train_seed,
        "map": map_name,
        "condition": condition,
        "seed": seed,
        "status": status,
        "total_time_s": time if status == "FINISHED" else None,
        "wall_collision_steps": 0,
    }


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


def test_provenance_pins_the_inputs_of_a_run():
    assert {"commit", "dirty", "sim_sha256", "features_sha256", "reward_sha256"} <= set(
        provenance()
    )


def test_times_are_only_compared_on_races_both_drivers_finished():
    records = [
        record("ppo-h1", 0, "easy", time=100.0),
        record("ppo-h4", 0, "easy", time=90.0),
        record("ppo-h1", 0, "hard", "DNF"),
        record("ppo-h4", 0, "hard", time=500.0),
    ]
    paired = paired_differences(records)
    assert paired["finish_rate"] == {0: 0.5}
    assert paired["time_gain_s"] == {0: 10.0}


def test_seeds_without_a_counterpart_are_left_out():
    records = [record("ppo-h1", 0, "a"), record("ppo-h4", 0, "a"), record("ppo-h4", 1, "a")]
    assert list(paired_differences(records)["finish_rate"]) == [0]


def test_no_shared_finish_gives_no_time_gain():
    records = [record("ppo-h1", 0, "a", "DNF"), record("ppo-h4", 0, "a")]
    assert paired_differences(records)["time_gain_s"] == {}


def test_too_few_seeds_give_no_interval():
    assert bootstrap_interval([1.0, 2.0]) is None


def test_interval_contains_the_mean_and_is_reproducible():
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    low, high = bootstrap_interval(values)
    assert low < 2.0 < high
    assert bootstrap_interval(values) == (low, high)


@pytest.mark.parametrize(
    ("finish", "gain", "expected"),
    [
        ((0.1, 0.3), None, "h4 plus fiable"),
        ((-0.3, -0.1), (1.0, 2.0), "h4 moins fiable"),
        ((-0.1, 0.1), (1.0, 2.0), "h4 plus rapide"),
        ((-0.1, 0.1), (-2.0, -1.0), "h4 plus lent"),
        ((-0.1, 0.1), (-1.0, 1.0), "non concluant"),
        ((-0.1, 0.1), None, "non concluant"),
        (None, (1.0, 2.0), "non concluant"),
    ],
)
def test_verdict_follows_the_rule_fixed_in_advance(finish, gain, expected):
    assert verdict(finish, gain) == expected


def test_summary_counts_failures_in_the_rate():
    summary = driver_summary(
        [record("d", 0, "a", time=10.0), record("d", 0, "b", "DNF"), record("d", 0, "c", "TIMEOUT")]
    )
    assert summary["finish_rate"] == pytest.approx(1 / 3)
    assert (summary["dnf"], summary["timeout"]) == (1, 1)
    assert summary["median_time_s"] == 10.0


def test_summary_without_any_finish_has_no_time():
    assert driver_summary([record("d", 0, "a", "DNF")])["median_time_s"] is None
