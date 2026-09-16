"""Contract of the vendored simulator as observed at import time (guards E-12 fixes)."""

import numpy as np

REAL_OPPONENT_KEYS = {"x_rel", "y_rel", "yaw_rel", "speed", "progress", "lap_count", "status"}
MAX_SLOTS = 4


def test_observation_schema(sim):
    obs = sim.get_obs(0)
    assert {"lidar", "velocity", "steering", "progress", "lap_count", "rank", "opponents"} <= set(
        obs
    )
    assert obs["lidar"].shape == (100,)
    assert obs["lidar"].dtype == np.float32
    assert np.all((obs["lidar"] >= 0.1) & (obs["lidar"] <= 15.0))
    assert set(obs["opponents"]) == set(range(MAX_SLOTS))
    for slot in obs["opponents"].values():
        assert set(slot) == REAL_OPPONENT_KEYS


def test_ego_and_absent_slots_are_neutral(sim):
    opponents = sim.get_obs(0)["opponents"]
    # slot 0 is the ego car, slots 2-3 are absent (fixture runs 2 cars)
    for slot in (0, 2, 3):
        values = opponents[slot]
        assert values["status"] == 0
        assert all(values[k] == 0 for k in REAL_OPPONENT_KEYS - {"status"})
    assert opponents[1]["status"] == 1


def test_step_info_has_no_done_flag(sim):
    info = sim.get_step_info()
    assert not {"done", "terminated", "truncated"} & set(info)
    for key in ("step_count", "time_elapsed", "collisions", "friction_current", "agent_status"):
        assert key in info


def test_actions_are_clamped_to_bounds(sim):
    space = sim.get_space_info()["actions"]
    assert space["target_speed"]["bounds"] == (-2.0, 10.0)
    assert np.allclose(space["steering"]["bounds"], (-0.4189, 0.4189), atol=1e-4)

    sim.apply_action(1, 99.0, 9.0)
    # ponytail: reads the private action buffer, the only place the clamp is observable
    steer, speed = sim._sim_instance._actions[1]
    assert speed == 10.0
    assert np.isclose(steer, space["steering"]["bounds"][1])
