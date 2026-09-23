"""Defects of the vendored simulator reproduced by execution (E-10, fixes tracked in E-12).

Each test describes the EXPECTED behavior (per the engine's own comments, INSTRUCTIONS.md, or the
subject). Defects still present are marked `xfail(strict=True)`: the day one is fixed, the test
flips to XPASS strict, which breaks the suite and forces the marker to be removed. E-12 fixed D1,
D4, D5 and D6 (see vendor/PROVENANCE.md); D2 and D3 are left to our own adapter (E-14/E-16).
"""

import importlib

import env_simulation as es
import numpy as np
import pytest
from conftest import pure_pursuit


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


def test_friction_changes_over_time():
    """D1, fixed in E-12: friction changes every 20 s -> over 30 s, at least one change."""
    es.reset(1)
    seen = set()
    for _ in range(600):
        es.apply_action(0, 2.0, 0.0)
        es.simulation_step()
        seen.add(es.get_step_info()["friction_current"])
    assert len(seen) > 1


def test_engine_mu_follows_friction_current():
    """D1, fixed in E-12: the engine's mu was never updated after reset."""
    es.reset(1)
    sim = es._get_sim()
    sim._friction_countdown = 1
    for _ in range(3):
        es.simulation_step()
    assert sim._sim.agents[0].params["mu"] == es.get_step_info()["friction_current"] < 1.0


def test_training_friction_profile_is_applied(monkeypatch):
    """E-12: training-only randomisation, separate from the shipped local rules."""
    for name in ("_FRICTION_LO", "_FRICTION_HI", "_FRICTION_INTERVAL_SEC"):
        monkeypatch.setattr(es, name, getattr(es, name))  # restored after the test
    es.set_friction_profile(0.5, 0.8, (1.0, 2.0))
    es.reset(1)
    sim = es._get_sim()
    assert es.get_step_info()["friction_current"] == es._FRICTION_HI == 0.8
    seen = set()
    for _ in range(200):  # 10 s: at least 5 draws
        es.simulation_step()
        mu = es.get_step_info()["friction_current"]
        assert 0.5 <= mu <= 0.8
        assert sim._sim.agents[0].params["mu"] == mu
        seen.add(mu)
    assert len(seen) >= 5


@pytest.mark.parametrize(
    "lo, hi, interval", [(0.0, 1.0, (20, 20)), (0.9, 0.8, (20, 20)), (0.9, 1.0, (0.0, 1.0))]
)
def test_training_friction_profile_rejects_invalid_values(lo, hi, interval):
    with pytest.raises(ValueError):
        es.set_friction_profile(lo, hi, interval)


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


def test_every_available_map_can_be_loaded():
    """D4, fixed in E-12: "Mexico City" was listed but its files are named MexicoCity_*, so
    set_map("Mexico City") failed. The duplicate folder is gone; every listed name loads
    (all 23 are driven in tests/test_simulator_maps.py)."""
    assert "Mexico City" not in es.get_available_maps()
    assert len(es.get_available_maps()) == 23


@pytest.fixture
def ghost_map(monkeypatch):
    """A listed map with no files, rather than "Mexico City": fixing D4 must not hide D5."""
    monkeypatch.setattr(es, "_available_maps", es._ensure_maps_discovered() | {"Ghost"})


def test_failed_set_map_does_not_poison_singleton(ghost_map):
    """D5, fixed in E-12: a failed set_map left _current_map on the broken map."""
    es.set_map("Spa")
    es.reset(1)
    arc = es._get_sim()._total_arc
    with pytest.raises(FileNotFoundError):
        es.set_map("Ghost")
    assert es.get_current_map() == "Spa"
    es.reset(1)
    assert es._get_sim()._total_arc == arc  # still projecting progress on Spa


def test_failed_set_map_before_any_reset_keeps_the_previous_map(ghost_map):
    """D5, fixed in E-12: with no simulator built yet, the failure surfaced at the next reset."""
    es.close()
    before = es.get_current_map()
    with pytest.raises(FileNotFoundError):
        es.set_map("Ghost")
    assert es.get_current_map() == before
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


# --- Lap-line stagnation DNF (found by the E-11 control driver) --------------------


def test_reversing_over_the_start_line_is_not_dnf():
    """E-12, bug 2 of docs/scripted_driver.md: backing over the line put the lap phase near
    0.99, the stagnation max stuck there and the car was DNF'd at step 81 while driving on."""
    es.set_map("IMS")
    es.reset(1)
    sim = es._get_sim()
    for _ in range(10):
        es.apply_action(0, -1.0, 0.0)
        es.simulation_step()
    for _ in range(120):  # straight on down the start straight, well past the 80-step window
        es.apply_action(0, 2.0, 0.0)
        es.simulation_step()
        info = es.get_step_info()
        assert info["collisions"]["wall"][0] == 0
        assert info["agent_status"][0] == 1, f"DNF at step {info['step_count']}"
    assert sim._cum[0] > 0.01  # it did make progress past its start


@pytest.mark.slow
def test_crossing_the_line_does_not_dnf():
    """E-12, bug 1 of docs/scripted_driver.md: on the line the projection is exactly 0.0 and
    (0.0 - p0) % 1.0 == 1.0 when p0 ~ 1e-19, so the phase hit 1.0 right after the lap was
    booked, the stagnation max stuck at 1.0 and the car was DNF'd 80 decisions later."""
    es.set_map("IMS")
    es.reset(1)
    sim = es._get_sim()
    after_line = 0
    while after_line < 120:
        es.apply_action(0, *pure_pursuit(sim, 0, speed=3.0))
        es.simulation_step()
        info = es.get_step_info()
        assert info["agent_status"][0] == 1, f"DNF at step {info['step_count']}"
        assert 0.0 <= es.get_obs(0)["progress"] < 1.0
        after_line += es.get_obs(0)["lap_count"] >= 1
        assert info["step_count"] < 3000, "no lap in 150 s"


# --- Parameter consistency (E-12) ------------------------------------------------


def test_physics_and_decision_rates_are_consistent():
    es.reset(1)
    sim = es._get_sim()
    assert es._PHYSICS_STEPS / es._FREQ_HZ == pytest.approx(1 / es._DECISION_FREQ_HZ)
    assert sim._sim.time_step == pytest.approx(1 / es._FREQ_HZ)
    assert all(agent.ttc_thresh == es._TTC_THRESH for agent in sim._sim.agents)


def test_engine_mu_is_the_friction_not_the_nominal_parameter():
    """Kept as shipped: reset() overrides the nominal mu (1.0489) with friction_current (1.0),
    so the nominal value never reaches the physics. Documented in vendor/PROVENANCE.md."""
    es.reset(1)
    mu = es._get_sim()._sim.agents[0].params["mu"]
    assert es._PARAMS["mu"] == 1.0489
    assert mu == es.get_step_info()["friction_current"] == es._FRICTION_HI


# --- Documentation vs code -------------------------------------------------------


def test_get_obs_docstring_overstates_lidar_noise():
    """Confirmed static observation: the docstring says +/-3%, the constant is +/-0.1%."""
    assert "3%" in es.get_obs.__doc__
    assert es._LIDAR_NOISE == 0.001
