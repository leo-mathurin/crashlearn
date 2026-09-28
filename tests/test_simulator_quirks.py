"""Official 2026-09-18 simulator behaviour an agent must live with or can exploit (E-10).

The simulator is vendored unchanged: it is what the tournament runs, and exploiting its quirks
is allowed. These tests pin what was measured, so that a new archive changing a rule breaks the
suite instead of silently changing what our agents learn. D1-D6 refer to the audit of the first
archive (docs/audit_simulateur.md).
"""

import importlib
import math
from pathlib import Path

import env_simulation as es
import numpy as np
import pytest
from conftest import pure_pursuit
from f110_gym.envs.base_classes import RaceCar


@pytest.fixture(autouse=True)
def _fresh_sim():
    """Close the singleton after each test and restore the example map (a module global)."""
    yield
    es.close()
    if es.get_current_map() not in (None, "example"):
        es.set_map("example")


# --- Friction (D1: still frozen upstream, left as a TODO) ------------------------


def test_friction_is_frozen_at_one():
    """Upstream leaves `_update_friction` as a TODO: varying it is our training-side job (E-23)."""
    es.reset(1)
    seen = set()
    for _ in range(600):
        es.apply_action(0, 2.0, 0.0)
        es.simulation_step()
        seen.add(es.get_step_info()["friction_current"])
    assert seen == {1.0}
    assert es._get_sim()._sim.agents[0].params["mu"] == 1.0


def test_friction_bounds_are_almost_flat():
    assert (es._FRICTION_LO, es._FRICTION_HI) == (0.99, 1.0)
    assert es._FRICTION_INTERVAL_SEC == (20, 20)


# --- Observation schemas (D2/D3 fixed upstream) -----------------------------------


def test_space_info_opponent_keys_match_real_obs():
    es.reset(2)
    real = set(es.get_obs(0)["opponents"][1])
    announced = es.get_space_info()["observations"]["opponents"]["unit"]
    for key in real:
        assert key in announced


def test_loader_dummy_obs_matches_real_obs_schema():
    agent_loader = importlib.import_module("agent_loader")
    es.reset(2)
    real = es.get_obs(0)
    dummy = agent_loader.create_dummy_obs()
    assert set(dummy) == set(real)
    assert set(dummy["opponents"][0]) == set(real["opponents"][1])


def test_loader_dummy_info_matches_real_info_schema():
    agent_loader = importlib.import_module("agent_loader")
    es.reset(1)
    real = es.get_step_info()
    dummy = agent_loader.create_dummy_info()
    assert set(dummy) == set(real)
    assert type(dummy["collisions"]["wall"]) is type(real["collisions"]["wall"])


# --- Maps (D4/D5 fixed upstream) --------------------------------------------------


def test_failed_set_map_does_not_poison_singleton(monkeypatch):
    monkeypatch.setattr(es, "_available_maps", es._ensure_maps_discovered() | {"Ghost"})
    es.reset(1)
    with pytest.raises(FileNotFoundError):
        es.set_map("Ghost")
    assert es.get_current_map() != "Ghost"
    es.reset(1)


# --- Walls (D6 closed upstream by the off-track DNF and the wall bounce) -----------


def _drive_into_nearest_wall(steps: int) -> list[dict]:
    """Point the car at the nearest wall and floor it; returns one record per step."""
    es.reset(1)
    sim = es._get_sim()
    scan = es.get_obs(0)["lidar"]
    st = sim._sim.agents[0].state
    st[4] += float(RaceCar.scan_angles[int(np.argmin(scan))])
    st[3] = 0.0
    x0, y0 = float(st[0]), float(st[1])
    records = []
    for _ in range(steps):
        es.apply_action(0, es._TARGET_SPEED_MAX, 0.0)
        es.simulation_step()
        st = sim._sim.agents[0].state
        status = es.get_step_info()["agent_status"][0]
        records.append({"moved": float(np.hypot(st[0] - x0, st[1] - y0)), "status": status})
        if status != 1:
            break
    return records


def test_pushing_into_a_thick_wall_ends_in_stagnation_dnf():
    """Measured on the example map: the car bounces, then creeps into the wall by ~5 mm per
    step (the front ray reads 15 m once inside) and is DNF'd once the 10 s window shows less than
    1 m of progress, before it comes out on the other side."""
    records = _drive_into_nearest_wall(400)
    active = [r for r in records if r["status"] == 1]
    assert max(r["moved"] for r in active) < 1.5
    assert records[-1]["status"] == 0
    assert len(records) == es._DNF_WINDOW_STEPS + 1 == 201


def test_car_outside_the_track_corridor_is_dnf_immediately():
    """The off-track DNF uses the local half-width plus _OFFTRACK_MARGIN_M from the centerline."""
    es.reset(1)
    sim = es._get_sim()
    st = sim._sim.agents[0].state
    st[0] += 200.0
    es.apply_action(0, 0.0, 0.0)
    es.simulation_step()
    assert es.get_step_info()["agent_status"][0] == 0


def test_offtrack_margin_differs_from_the_instructions():
    """The code runs with 1.00 m; INSTRUCTIONS.md announces 0.50 m (question asked upstream)."""
    assert es._OFFTRACK_MARGIN_M == 1.0
    instructions = (Path(es.__file__).parent / "INSTRUCTIONS.md").read_text()
    assert "0.50 m tolerance" in instructions


@pytest.mark.slow
def test_a_thin_wall_can_be_cut_through_for_free_progress():
    """Measured on Zandvoort: from the start point, aiming at centerline point 174 (2.66 m away
    across the wall, 78 m further along the track) at full speed, the car creeps through the wall
    and is credited ~78 m of progress while staying ACTIVE: the corridor is narrow enough that
    the off-track DNF never fires. Thicker walls end in a stagnation DNF instead."""
    es.set_map("Zandvoort")
    es.reset(1)
    sim = es._get_sim()
    w = sim._waypoints
    target = w[174]
    st = sim._sim.agents[0].state
    st[0], st[1] = w[0]
    st[3] = 0.0
    st[4] = math.atan2(target[1] - w[0, 1], target[0] - w[0, 0])
    es.simulation_step()
    start = float(sim._cum[0])
    for _ in range(300):
        st = sim._sim.agents[0].state
        if math.hypot(st[0] - target[0], st[1] - target[1]) < 0.4:
            break
        heading = math.atan2(target[1] - st[1], target[0] - st[0]) - st[4]
        steer = math.atan2(math.sin(heading), math.cos(heading))
        es.apply_action(0, es._TARGET_SPEED_MAX, steer)
        es.simulation_step()
    assert es.get_step_info()["agent_status"][0] == 1
    assert (float(sim._cum[0]) - start) * sim._total_arc > 70.0


def test_reverse_frees_car_after_long_wall_push():
    _drive_into_nearest_wall(120)
    sim = es._get_sim()
    for _ in range(60):
        es.apply_action(0, es._TARGET_SPEED_MIN, 0.0)
        es.simulation_step()
        if float(sim._sim.agents[0].state[3]) < -0.1:
            return
    pytest.fail("speed is still zero after 60 reverse steps")


# --- Vehicle contacts: being rear-ended is a free push forward ----------------------


def _rear_end(ram: bool, steps: int = 100) -> tuple[float, float, int]:
    """Car 1 starts 1 m behind car 0 on Spa; car 0 drives at 6 m/s, car 1 at 10 m/s if `ram`."""
    es.set_map("Spa")
    es.reset(2)
    sim = es._get_sim()
    front, rear = sim._sim.agents
    rear.state[:] = front.state
    rear.state[0] -= math.cos(front.state[4])
    rear.state[1] -= math.sin(front.state[4])
    es.simulation_step()
    start = sim._cum.copy()
    contacts = 0
    for _ in range(steps):
        es.apply_action(0, *pure_pursuit(sim, 0, speed=6.0))
        es.apply_action(1, *pure_pursuit(sim, 1, speed=10.0 if ram else 0.0))
        es.simulation_step()
        info = es.get_step_info()
        contacts += bool(info["collisions"]["vehicle"][0])
        assert info["agent_status"][0] == info["agent_status"][1] == 1
    gained = (sim._cum - start) * sim._total_arc
    return float(gained[0]), float(gained[1]), contacts


@pytest.mark.slow
def test_being_rear_ended_pushes_the_front_car_forward():
    """Measured on Spa over 5 s: ~+4.5 m for the front car, no DNF, no damage."""
    alone, _, _ = _rear_end(ram=False)
    pushed, _, contacts = _rear_end(ram=True)
    assert contacts > 0
    assert pushed - alone > 1.0


def test_knockback_constants():
    assert es._KNOCKBACK_REST == 0.15
    assert es._KNOCKBACK_MAX_M == 0.5
    assert es._KNOCKBACK_MIN_SHARE == 0.2
    assert (es._WALL_REST, es._WALL_REST_MAX) == (0.3, 0.45)
