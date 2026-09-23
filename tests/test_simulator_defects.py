"""Defects of the vendored simulator reproduced by execution (E-10, fixes tracked in E-12).

Each test describes the EXPECTED behavior (per the engine's own comments, INSTRUCTIONS.md, or the
subject) and is marked `xfail(strict=True)`: it fails today, and the day E-12 fixes the underlying
defect it will flip to XPASS strict, which breaks the suite and forces the marker to be removed.
"""

import importlib

import env_simulation as es
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _fresh_sim():
    """Close the singleton after each test and restore the example map.

    `set_map` keeps its state in module globals; some defects below (D4/D5) leave that state
    poisoned on failure, so the map is restored explicitly to avoid contaminating later tests.
    """
    yield
    es.close()
    if es.get_current_map() not in (None, "example"):
        es.set_map("example")


# --- Friction -----------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="D1: _update_friction is a no-op, friction stays at 1.0")
def test_friction_changes_over_time():
    """Announced friction changes every 20s -> over 30s, at least one change is expected."""
    es.reset(1)
    seen = set()
    for _ in range(600):
        es.apply_action(0, 2.0, 0.0)
        es.simulation_step()
        seen.add(es.get_step_info()["friction_current"])
    assert len(seen) > 1


@pytest.mark.xfail(strict=True, reason="D1: the engine's mu is never updated after reset")
def test_engine_mu_follows_friction_current():
    es.reset(1)
    sim = es._get_sim()
    sim._friction_countdown = 1
    for _ in range(3):
        es.simulation_step()
    assert sim._sim.agents[0].params["mu"] == es.get_step_info()["friction_current"] < 1.0


def test_friction_bounds_are_almost_flat():
    """Confirmed static observation: the announced [0.99, 1.0] range is nowhere near rain at 0.5."""
    assert (es._FRICTION_LO, es._FRICTION_HI) == (0.99, 1.0)
    assert es._FRICTION_INTERVAL_SEC == (20, 20)


# --- Inconsistent observation schemas ------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="D2: get_space_info announces rel_x/rel_y/.../active, get_obs returns x_rel/.../status",
)
def test_space_info_opponent_keys_match_real_obs():
    es.reset(2)
    real = set(es.get_obs(0)["opponents"][1])
    announced = es.get_space_info()["observations"]["opponents"]["unit"]
    for key in real:
        assert key in announced


@pytest.mark.xfail(
    strict=True,
    reason="D3: create_dummy_obs has neither agent_id nor the real opponent keys",
)
def test_loader_dummy_obs_matches_real_obs_schema():
    agent_loader = importlib.import_module("agent_loader")
    es.reset(2)
    real = es.get_obs(0)
    dummy = agent_loader.create_dummy_obs()
    assert set(dummy) == set(real)
    assert set(dummy["opponents"][0]) == set(real["opponents"][1])


@pytest.mark.xfail(
    strict=True,
    reason="D3: create_dummy_info is missing time_elapsed/ranks/progress_delta/race_over "
    "and uses lists instead of dicts",
)
def test_loader_dummy_info_matches_real_info_schema():
    agent_loader = importlib.import_module("agent_loader")
    es.reset(1)
    real = es.get_step_info()
    dummy = agent_loader.create_dummy_info()
    assert set(dummy) == set(real)
    assert type(dummy["collisions"]["wall"]) is type(real["collisions"]["wall"])


# --- Maps ---------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    raises=FileNotFoundError,
    reason="D4: 'Mexico City' is listed but its files are named MexicoCity_*",
)
def test_every_available_map_can_be_loaded():
    assert "Mexico City" in es.get_available_maps()
    es.set_map("Mexico City")
    es.reset(1)


@pytest.mark.xfail(
    strict=True,
    reason="D5: a failed set_map leaves _current_map on the broken map, reset() then crashes",
)
def test_failed_set_map_does_not_poison_singleton(monkeypatch):
    # A listed map with no files, rather than "Mexico City": fixing D4 must not hide D5.
    monkeypatch.setattr(es, "_available_maps", es._ensure_maps_discovered() | {"Ghost"})
    es.reset(1)
    with pytest.raises(FileNotFoundError):
        es.set_map("Ghost")
    # Expected: the previous state is kept and reset() still works.
    assert es.get_current_map() != "Ghost"
    es.reset(1)


# --- Passable walls -------------------------------------------------------------


def _drive_into_nearest_wall(steps: int) -> float:
    """Point the car at the nearest wall and floor it.

    Returns the farthest distance from the start reached while the car was ACTIVE: a car
    pinned against a wall is DNF'd by the 4 s stagnation rule and teleported off-map, which
    is expected and must not count as going through the wall. Every active pose must also
    leave the body in free space.
    """
    from f110_gym.envs.base_classes import RaceCar

    es.reset(1)
    sim = es._get_sim()
    scan = es.get_obs(0)["lidar"]
    st = sim._sim.agents[0].state
    st[4] += float(RaceCar.scan_angles[int(np.argmin(scan))])
    st[3] = 0.0
    x0, y0 = float(st[0]), float(st[1])
    farthest = 0.0
    for _ in range(steps):
        es.apply_action(0, es._TARGET_SPEED_MAX, 0.0)
        es.simulation_step()
        if es.get_step_info()["agent_status"][0] != 1:
            break
        st = sim._sim.agents[0].state
        assert es._footprint_clear(st), "an active car ended a step inside a wall"
        farthest = max(farthest, float(np.hypot(st[0] - x0, st[1] - y0)))
    return farthest


def test_wall_is_impassable_when_pushing_for_10_seconds():
    """D6, fixed in E-12: before the fix the car crept then tunneled (133 m off-map in 400
    steps, still ACTIVE) because the iTTC check never fires at v~=0 nor inside a wall."""
    moved = _drive_into_nearest_wall(200)
    assert moved < 1.0, f"car went through the wall: {moved:.1f} m traveled"


def test_car_pinned_against_a_wall_is_dnf_by_stagnation():
    """D6, fixed in E-12: a car that tunneled kept progressing and was never DNF'd."""
    _drive_into_nearest_wall(400)
    assert es.get_step_info()["agent_status"][0] == 0


def test_reverse_frees_car_after_wall_push():
    """D6, fixed in E-12: reversing did not free a car that had penetrated the wall.
    The push lasts 3 s, under the 4 s stagnation window (the car must still be ACTIVE)."""
    _drive_into_nearest_wall(60)
    assert es.get_step_info()["agent_status"][0] == 1
    sim = es._get_sim()
    for _ in range(20):
        es.apply_action(0, es._TARGET_SPEED_MIN, 0.0)
        es.simulation_step()
        if float(sim._sim.agents[0].state[3]) < -0.1:
            return
    pytest.fail("speed is still zero after 20 reverse steps")


# --- Documentation vs code -------------------------------------------------------


def test_get_obs_docstring_overstates_lidar_noise():
    """Confirmed static observation: the docstring says +/-3%, the constant is +/-0.1%."""
    assert "3%" in es.get_obs.__doc__
    assert es._LIDAR_NOISE == 0.001
