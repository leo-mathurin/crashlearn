"""Measured dynamic behavior (E-10): LiDAR, DNF, walls, laps, finish, ranking."""

import env_simulation as es
import numpy as np
import pytest
from conftest import pure_pursuit
from f110_gym.envs.base_classes import RaceCar


@pytest.fixture(scope="module", autouse=True)
def _restore_shared_sim_state():
    """This module never touches the current map, but leaves the singleton mid-episode with an
    arbitrary car count; restore the 2-car state `conftest.sim` hands to later modules (notably
    `test_submission_contract.py`, which reuses that session-scoped fixture as-is)."""
    yield
    es.reset(2)
    es.simulation_step()


# --- LiDAR ------------------------------------------------------------------


def _raw_scan(cid: int) -> np.ndarray:
    ag = es._get_sim()._sim.agents[cid]
    pose = np.array([ag.state[0], ag.state[1], ag.state[4]])
    return RaceCar.scan_simulator.scan(pose, None)


def test_lidar_noise_is_multiplicative_and_tiny():
    """Real noise: uniform +/-0.1% (the _LIDAR_NOISE constant), not the announced +/-3/5%."""
    es.reset(1)
    es.simulation_step()
    raw = _raw_scan(0)
    obs = es.get_obs(0)["lidar"]
    inside = (raw > es._LIDAR_MIN) & (raw < es._LIDAR_MAX)
    rel = np.abs(obs[inside] / raw[inside] - 1.0)
    assert rel.max() <= es._LIDAR_NOISE + 1e-6
    assert rel.max() > 1e-5, "some noise is indeed applied"
    assert es._LIDAR_NOISE == 0.001


def test_lidar_has_no_dropout():
    es.reset(1)
    for _ in range(50):
        es.apply_action(0, 3.0, 0.0)
        es.simulation_step()
        scan = es.get_obs(0)["lidar"]
        raw = _raw_scan(0)
        # No ray is zeroed/clamped to min while the geometry returns a real distance.
        assert not np.any((scan <= es._LIDAR_MIN) & (raw > 0.5))


def test_lidar_is_stable_between_steps_and_noise_is_frozen_per_step():
    es.reset(1)
    es.simulation_step()
    a = es.get_obs(0)["lidar"]
    b = es.get_obs(0)["lidar"]
    assert np.array_equal(a, b), "get_obs reuses the noise drawn at the last simulation_step"


def test_lidar_sees_walls_not_cars():
    """Car 0's scan is identical whether car 1 sits right next to it or far away."""
    es.reset(2)
    es.simulation_step()
    scan_with_neighbour = es.get_obs(0)["lidar"].copy()
    # Physically move car 1 away without moving car 0 or clearing any noise.
    es._get_sim()._sim.agents[1].state[0] += 50.0
    scan_without_neighbour = es.get_obs(0)["lidar"]
    assert np.array_equal(scan_with_neighbour, scan_without_neighbour)


# --- Anti-stagnation / DNF ----------------------------------------------------


def test_idle_car_is_dnf_after_window_plus_one_steps():
    """An idle car is DNF'd at exactly step 81 (80-step window = 4s at 20 Hz)."""
    es.reset(1)
    dnf_step = None
    for k in range(1, 200):
        es.apply_action(0, 0.0, 0.0)
        es.simulation_step()
        info = es.get_step_info()
        if info["agent_status"][0] == 0:
            dnf_step = k
            break
    assert dnf_step == es._DNF_WINDOW_STEPS + 1 == 81
    assert info["race_over"] is True
    assert info["ranks"][0] == 1, "sole car: rank 1 even while DNF"


def test_dnf_car_is_teleported_off_track_and_frozen():
    es.reset(2)
    for _ in range(es._DNF_WINDOW_STEPS + 1):
        es.apply_action(0, 3.0, 0.0)
        es.apply_action(1, 0.0, 0.0)
        es.simulation_step()
    info = es.get_step_info()
    assert info["agent_status"] == {0: 1, 1: 0, 2: 0, 3: 0}
    st1 = es._get_sim()._sim.agents[1].state
    assert st1[0] > 1000 and st1[1] > 1000
    # Seen from car 0: slot 1 is neutralized, status 0.
    assert es.get_obs(0)["opponents"][1]["status"] == 0
    assert es.get_obs(0)["opponents"][1]["x_rel"] == 0.0
    assert info["ranks"][1] == 2


def test_moving_car_is_not_dnf_over_the_window():
    es.reset(1)
    for _ in range(es._DNF_WINDOW_STEPS + 20):
        es.apply_action(0, 3.0, 0.0)
        es.simulation_step()
    assert es.get_step_info()["agent_status"][0] == 1


# --- Walls ---------------------------------------------------------------------


def _face_nearest_wall(cid: int = 0) -> None:
    scan = es.get_obs(cid)["lidar"]
    st = es._get_sim()._sim.agents[cid].state
    st[4] = float(st[4]) + float(RaceCar.scan_angles[int(np.argmin(scan))])
    st[3] = 0.0


def _state(cid: int = 0) -> np.ndarray:
    """The engine reassigns `agent.state` on every sub-step (`base_classes.py:378`): never hold
    a reference across two `simulation_step()` calls."""
    return es._get_sim()._sim.agents[cid].state


def test_wall_contact_zeroes_speed_keeps_yaw_and_does_not_terminate():
    """First wall contact: flag raised, speed zeroed, heading kept, status stays ACTIVE."""
    es.reset(1)
    _face_nearest_wall(0)
    hit_at = None
    for k in range(80):
        yaw_before = float(_state()[4])
        es.apply_action(0, es._TARGET_SPEED_MAX, 0.0)
        es.simulation_step()
        info = es.get_step_info()
        if info["collisions"]["wall"][0]:
            hit_at = k
            assert info["agent_status"][0] == 1
            assert float(_state()[3]) == 0.0
            assert float(_state()[4]) == pytest.approx(yaw_before), "yaw is not clobbered to 0"
            break
    assert hit_at is not None and hit_at < 20


def test_reverse_frees_car_shortly_after_wall_contact():
    """Measured: after 10 steps of pushing, reversing frees the car within <=10 steps (see the
    D6 defect tests for the case where the car has already penetrated the wall)."""
    es.reset(1)
    _face_nearest_wall(0)
    while not es.get_step_info()["collisions"]["wall"][0]:
        es.apply_action(0, es._TARGET_SPEED_MAX, 0.0)
        es.simulation_step()
    for _ in range(10):
        es.apply_action(0, es._TARGET_SPEED_MAX, 0.0)
        es.simulation_step()
    freed = None
    for r in range(20):
        es.apply_action(0, es._TARGET_SPEED_MIN, 0.0)
        es.simulation_step()
        if not es.get_step_info()["collisions"]["wall"][0] and float(_state()[3]) < -0.1:
            freed = r
            break
    assert freed is not None


# --- Laps, finish, ranking ------------------------------------------------


@pytest.mark.slow
def test_pure_pursuit_completes_three_laps_and_finishes():
    """Full race on the example circuit: 3 laps -> status 2, race_over, 3 lap_times."""
    es.reset(1)
    sim = es._get_sim()
    lap_steps = []
    final = None
    for k in range(1, 6000):
        es.apply_action(0, *pure_pursuit(sim, 0, speed=3.0))
        es.simulation_step()
        info = es.get_step_info()
        if info["lap_complete"][0]:
            lap_steps.append(k)
            assert es.get_obs(0)["lap_count"] == len(lap_steps)
        if info["agent_status"][0] != 1:
            final = k
            break
    assert final is not None, "the car never finished"
    assert info["agent_status"][0] == 2
    assert info["race_over"] is True
    assert len(info["lap_times"][0]) == es._REQUIRED_LAPS == 3
    assert all(t > 0 for t in info["lap_times"][0])
    # The last lap's lap_complete is masked because the car is no longer active (returned as
    # np.bool_ rather than a Python bool in that case: see O3 in the audit).
    assert len(lap_steps) == es._REQUIRED_LAPS - 1
    assert not info["lap_complete"][0]
    # Order of magnitude: ~52s per lap at 3 m/s over 156 m (measured: 1045/2084/3124).
    assert 900 < lap_steps[0] < 1200
    assert 2800 < final < 3400


@pytest.mark.slow
def test_finished_car_ranks_first_and_is_removed():
    es.reset(2)
    sim = es._get_sim()
    for _ in range(3500):
        es.apply_action(0, *pure_pursuit(sim, 0, speed=3.0))
        es.apply_action(1, *pure_pursuit(sim, 1, speed=2.0))
        es.simulation_step()
        info = es.get_step_info()
        if info["agent_status"][0] == 2:
            break
    assert info["agent_status"][0] == 2
    assert info["agent_status"][1] == 1
    assert info["ranks"] == {0: 1, 1: 2, 2: 4, 3: 4}
    assert es.get_obs(1)["opponents"][0]["status"] == 2
    st0 = sim._sim.agents[0].state
    assert st0[0] > 1000, "finished car is teleported off track"


def test_ranks_follow_cumulative_progress_for_active_cars():
    es.reset(3)
    sim = es._get_sim()
    for _ in range(40):
        es.apply_action(0, 2.0, 0.0)
        es.apply_action(1, 4.0, 0.0)
        es.apply_action(2, 1.0, 0.0)
        es.simulation_step()
    info = es.get_step_info()
    order = sorted(range(3), key=lambda i: -sim._cum[i])
    assert [info["ranks"][i] for i in order] == [1, 2, 3]
    assert info["ranks"][3] == es._MAX_SLOTS, "absent slot -> rank _MAX_SLOTS"
