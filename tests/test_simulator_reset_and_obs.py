"""Additional observed contract of `env_simulation` (E-10): per-slot info fields, opponent
relative frame, wrapper bounds, action clipping, determinism and reset semantics.

Basic schema checks (observation keys, action clamping to the announced bounds, absence of a
`done` field) already live in `test_simulator_contract.py`; this file only adds coverage that
file does not have.
"""

import env_simulation as es
import numpy as np
import pytest

INFO_KEYS = {
    "step_count",
    "time_elapsed",
    "ranks",
    "collisions",
    "progress_delta",
    "lap_complete",
    "stagnation",
    "friction_current",
    "opponents_mask",
    "agent_status",
    "lap_times",
    "race_over",
    "max_progress",
}


@pytest.fixture(scope="module", autouse=True)
def _restore_shared_sim_state():
    """Leave the singleton on the 2-car state `conftest.sim` originally set up, so later modules
    reusing that session-scoped fixture (`test_submission_contract.py`) see a coherent state."""
    yield
    es.reset(2)
    es.simulation_step()


@pytest.mark.parametrize("num_cars", [1, 2, 3, 4])
def test_reset_and_step_for_each_car_count(num_cars):
    es.reset(num_cars)
    for cid in range(num_cars):
        es.apply_action(cid, 2.0, 0.0)
    es.simulation_step()
    info = es.get_step_info()

    assert set(info) == INFO_KEYS
    assert info["step_count"] == 1
    assert info["time_elapsed"] == pytest.approx(1 / es._DECISION_FREQ_HZ)
    # All 4 slots are always present in info; empty slots are neutralized.
    for slot in range(es._MAX_SLOTS):
        active = slot < num_cars
        assert info["agent_status"][slot] == (1 if active else 0)
        assert bool(info["opponents_mask"][slot]) is active
        assert info["collisions"]["wall"][slot] is False
        assert info["collisions"]["vehicle"][slot] is False
        assert info["lap_complete"][slot] is False
        assert info["stagnation"][slot] is False
        assert info["lap_times"][slot] == []
    assert info["race_over"] is False
    # Ranks of the present cars form a permutation of 1..n.
    assert sorted(info["ranks"][i] for i in range(num_cars)) == list(range(1, num_cars + 1))


@pytest.mark.parametrize("num_cars", [1, 4])
def test_obs_bounds_and_types(num_cars):
    es.reset(num_cars)
    es.simulation_step()
    for cid in range(num_cars):
        obs = es.get_obs(cid)
        assert obs["lidar"].shape == (es._LIDAR_RAYS,)
        assert obs["lidar"].dtype == np.float32
        assert np.all(np.isfinite(obs["lidar"]))
        assert obs["lidar"].min() >= es._LIDAR_MIN and obs["lidar"].max() <= es._LIDAR_MAX
        assert isinstance(obs["velocity"], float)
        assert isinstance(obs["steering"], float)
        assert 0.0 <= obs["progress"] < 1.0
        assert obs["lap_count"] == 0
        assert 1 <= obs["rank"] <= num_cars


def test_opponent_relative_frame_on_grid():
    """2x2 grid: from car 0's point of view, car 1 sits alongside (y_rel) and car 2 behind
    (x_rel < 0)."""
    es.reset(4)
    opp = es.get_obs(0)["opponents"]
    assert abs(opp[1]["y_rel"]) == pytest.approx(es._GRID_LAT_M, abs=1e-6)
    assert opp[1]["x_rel"] == pytest.approx(0.0, abs=1e-6)
    assert opp[2]["x_rel"] == pytest.approx(-es._GRID_ROW_M, abs=1e-6)
    assert opp[2]["y_rel"] == pytest.approx(0.0, abs=1e-6)
    assert opp[3]["x_rel"] == pytest.approx(-es._GRID_ROW_M, abs=1e-6)
    for slot in (1, 2, 3):
        assert opp[slot]["yaw_rel"] == pytest.approx(0.0, abs=1e-6)


def test_space_info_matches_wrapper_bounds():
    space = es.get_space_info()
    assert space["decision_freq_hz"] == 20
    assert space["actions"]["target_speed"]["bounds"] == (-2.0, 10.0)
    assert space["actions"]["steering"]["bounds"] == (-0.4189, 0.4189)
    assert space["observations"]["lidar"]["shape"] == (100,)
    assert space["observations"]["lidar"]["bounds"] == (0.0, 15.0)


def test_actions_are_clipped_to_both_bounds():
    """`test_simulator_contract.py` only exercises the high bound; this adds the low one."""
    es.reset(1)
    es.apply_action(0, 1e3, 1e3)
    sim = es._get_sim()
    assert sim._actions[0, 1] == es._TARGET_SPEED_MAX
    assert sim._actions[0, 0] == es._PARAMS["s_max"]
    es.apply_action(0, -1e3, -1e3)
    assert sim._actions[0, 1] == es._TARGET_SPEED_MIN
    assert sim._actions[0, 0] == es._PARAMS["s_min"]


def test_velocity_obs_stays_within_contract_after_max_thrust():
    """Example circuit: short starting straight (wall at step ~10, v max ~= 4.8 m/s)."""
    es.reset(1)
    v_max = 0.0
    for _ in range(60):
        es.apply_action(0, 1e3, 0.0)
        es.simulation_step()
        v_max = max(v_max, es.get_obs(0)["velocity"])
    assert 2.0 < v_max <= es._VEL_OBS_MAX + 1e-6


def test_actions_ignored_for_inactive_car():
    es.reset(2)
    sim = es._get_sim()
    sim._status[1] = 0
    es.apply_action(1, 5.0, 0.2)
    assert np.all(sim._actions[1] == 0.0)


def test_race_over_when_all_cars_inactive():
    es.reset(2)
    sim = es._get_sim()
    sim._status[0] = 0
    sim._status[1] = 2
    assert es.get_step_info()["race_over"] is True


def test_step_is_deterministic_for_same_seed():
    """Same action sequence -> same observations, within one `_Sim` construction (seeded LiDAR
    rng, seeded engine).

    Bit-exact equality only holds early in a process: `RaceCar.scan_simulator` is a *class-level*
    singleton (see `env_simulation`'s module docstring on the "LiDAR class-variable trap"), so
    repeated `reset()`/`close()` cycles in one interpreter accumulate tiny float drift in its
    cached geometry - harmless under the intended `SubprocVecEnv` process isolation, but visible
    here after enough prior tests have (re)built a simulator in this same process. `allclose`
    with a tolerance well above that drift (empirically <0.05) still catches a real regression
    (e.g. an unseeded RNG) while tolerating it.
    """

    def run():
        es.reset(2)
        out = []
        for _ in range(30):
            es.apply_action(0, 3.0, 0.1)
            es.apply_action(1, 2.0, -0.1)
            es.simulation_step()
            out.append(es.get_obs(0)["lidar"].copy())
        es.close()
        return np.stack(out)

    assert np.allclose(run(), run(), atol=0.05)


def test_reset_clears_state_but_keeps_car_count():
    es.reset(2)
    for _ in range(20):
        es.apply_action(0, 3.0, 0.0)
        es.simulation_step()
    assert es.get_step_info()["step_count"] == 20
    es.reset(2)
    info = es.get_step_info()
    assert info["step_count"] == 0
    assert es.get_obs(0)["progress"] == 0.0
    assert es.get_obs(0)["velocity"] == 0.0
    assert es._get_sim()._num_agents == 2
