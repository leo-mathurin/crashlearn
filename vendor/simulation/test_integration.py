"""
test_integration.py — Slow, physics-driven, map-dependent tests.
Run with: python test_integration.py
"""
import math
import numpy as np

from _harness import run_all, run_test, ok, fail

# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------
from env_simulation import (
    get_space_info, reset, get_obs, apply_action, simulation_step, get_step_info, close,
    _LIDAR_RAYS, _MAX_SLOTS, _get_sim, _DNF_WINDOW_STEPS, _FRICTION_HI, _PARAMS,
)
from f110_gym.envs.base_classes import RaceCar

# ---------------------------------------------------------------------------
# T05 — vehicle-vehicle collision: no termination, knockback visible
# ---------------------------------------------------------------------------
def t05_vehicle_collision_no_terminate_knockback():
    """
    Head-on: 2 cars launched at each other, the most violent contact there is
    (relative speed is doubled, so the impulse saturates _KNOCKBACK_MAX_M).
    Verify:
    - collision flag fires on both cars
    - episode does NOT terminate (no done, no status change)
    - the cars separate instead of interpenetrating
    - each car is pushed back along its OWN heading, i.e. in opposite directions
    """
    reset(2)
    sim_obj = _get_sim()
    assert sim_obj._sim is not None
    engine = sim_obj._sim

    # Override start positions: face each other ~1.2 m apart.
    # Car 0 keeps its (legal, on-track) start pose; car 1 is placed 1.2 m ahead
    # along car 0's heading, facing back. Placing them at arbitrary map coords
    # would land off the track edge and trigger the off-track DNF before contact.
    x0 = float(engine.agents[0].state[0])
    y0 = float(engine.agents[0].state[1])
    yaw0 = float(engine.agents[0].state[4])

    engine.agents[0].state[:] = 0.0
    engine.agents[0].state[0] = x0
    engine.agents[0].state[1] = y0
    engine.agents[0].state[4] = yaw0

    engine.agents[1].state[:] = 0.0
    engine.agents[1].state[0] = x0 + 1.2 * math.cos(yaw0)
    engine.agents[1].state[1] = y0 + 1.2 * math.sin(yaw0)
    engine.agents[1].state[4] = yaw0 + math.pi  # facing car 0

    # Unit vector along car 0's heading; car 1 faces exactly the other way.
    nx, ny = math.cos(yaw0), math.sin(yaw0)

    collision_detected = False
    # Seeded with the real pre-step gap so the first iteration compares against a
    # number, not None (a contact on step 0 would otherwise raise TypeError).
    prev_0 = (x0, y0)
    prev_1 = (float(engine.agents[1].state[0]), float(engine.agents[1].state[1]))
    dist_prev = math.hypot(prev_1[0] - prev_0[0], prev_1[1] - prev_0[1])

    for _ in range(60):
        apply_action(0, _PARAMS["v_max"], 0.0)
        apply_action(1, _PARAMS["v_max"], 0.0)
        simulation_step()
        info = get_step_info()

        x0 = float(engine.agents[0].state[0])
        y0 = float(engine.agents[0].state[1])
        x1 = float(engine.agents[1].state[0])
        y1 = float(engine.agents[1].state[1])
        dist = math.hypot(x1 - x0, y1 - y0)

        if info["collisions"]["vehicle"][0] and not collision_detected:
            collision_detected = True
            assert info["collisions"]["vehicle"][1], \
                "a head-on contact must be flagged on both cars"
            # No termination: status must still be ACTIVE
            assert info["agent_status"][0] == 1, "car 0 must be ACTIVE after collision"
            assert info["agent_status"][1] == 1, "car 1 must be ACTIVE after collision"
            assert dist >= dist_prev, (
                f"cars interpenetrating after knockback: at_hit={dist_prev:.4f}, after={dist:.4f}"
            )
            # Each car must be shoved backwards along its own heading. Both were
            # driving forward, so without a knockback each advance stays positive.
            adv_0 = (x0 - prev_0[0]) * nx + (y0 - prev_0[1]) * ny
            adv_1 = -((x1 - prev_1[0]) * nx + (y1 - prev_1[1]) * ny)
            assert adv_0 < 0.0 and adv_1 < 0.0, (
                f"cars not bounced in opposite directions: car 0 advanced {adv_0:+.4f} m, "
                f"car 1 advanced {adv_1:+.4f} m over the contact step"
            )
            break

        dist_prev = dist
        prev_0 = (x0, y0)
        prev_1 = (x1, y1)

    assert collision_detected, "vehicle-vehicle collision not detected"


# ---------------------------------------------------------------------------
# T05b — rear-end on a stopped car: the stopped car is shoved, the rammer pays too
# ---------------------------------------------------------------------------
def t05b_collision_with_stopped_car():
    """
    Car 1 drives head-on into car 0, which stays at zero throttle.
    Verify:
    - collision flag fires on BOTH cars
    - no termination (both still ACTIVE)
    - the stopped car is actually pushed away from the impact
    - the rammer does not walk through untouched: distance never collapses
    """
    reset(2)
    sim_obj = _get_sim()
    assert sim_obj._sim is not None
    engine = sim_obj._sim

    # Same on-track placement as T05: car 0 keeps its legal start pose,
    # car 1 sits 1.2 m ahead along car 0's heading, facing back.
    x0 = float(engine.agents[0].state[0])
    y0 = float(engine.agents[0].state[1])
    yaw0 = float(engine.agents[0].state[4])

    engine.agents[0].state[:] = 0.0
    engine.agents[0].state[0] = x0
    engine.agents[0].state[1] = y0
    engine.agents[0].state[4] = yaw0

    engine.agents[1].state[:] = 0.0
    engine.agents[1].state[0] = x0 + 1.2 * math.cos(yaw0)
    engine.agents[1].state[1] = y0 + 1.2 * math.sin(yaw0)
    engine.agents[1].state[4] = yaw0 + math.pi

    # Impact direction, seen from car 0: it must be shoved backwards (-heading).
    nx, ny = math.cos(yaw0), math.sin(yaw0)
    x0_init, y0_init = x0, y0

    collision_detected = False
    dist_prev = math.hypot(
        float(engine.agents[1].state[0]) - x0,
        float(engine.agents[1].state[1]) - y0,
    )
    # Per-step advance of the rammer along its own heading (= -n). Free of contact
    # it accelerates, so this grows monotonically; the impact must break that.
    prev_b = (float(engine.agents[1].state[0]), float(engine.agents[1].state[1]))
    advance_prev = 0.0

    for _ in range(60):
        apply_action(0, 0.0, 0.0)              # stopped car: no throttle
        apply_action(1, _PARAMS["v_max"], 0.0)  # rammer
        simulation_step()
        info = get_step_info()

        xa = float(engine.agents[0].state[0])
        ya = float(engine.agents[0].state[1])
        xb = float(engine.agents[1].state[0])
        yb = float(engine.agents[1].state[1])
        dist = math.hypot(xb - xa, yb - ya)
        advance = -((xb - prev_b[0]) * nx + (yb - prev_b[1]) * ny)

        if info["collisions"]["vehicle"][0] and not collision_detected:
            collision_detected = True
            assert info["collisions"]["vehicle"][1], \
                "collision must be flagged on both cars, not only on the rammer"
            assert info["agent_status"][0] == 1, "stopped car must stay ACTIVE"
            assert info["agent_status"][1] == 1, "rammer must stay ACTIVE"
            assert dist >= dist_prev, (
                f"cars interpenetrating after knockback: at_hit={dist_prev:.4f}, after={dist:.4f}"
            )
            # The stopped car must have been pushed back along -heading.
            push = (xa - x0_init) * nx + (ya - y0_init) * ny
            assert push < -1e-3, f"stopped car was not shoved backwards (push={push:.4f} m)"
            # The rammer must pay too: without a floor on its share of the contact,
            # ramming a stopped car costs it nothing and it keeps accelerating
            # straight through. Its advance over the contact step must drop.
            assert advance < advance_prev, (
                f"rammer went through untouched: advance {advance_prev:.4f} m before "
                f"contact, {advance:.4f} m during contact"
            )
            break

        dist_prev = dist
        prev_b = (xb, yb)
        advance_prev = advance

    assert collision_detected, "collision with a stopped car not detected"


# ---------------------------------------------------------------------------
# T10 — DNF'd agent doesn't corrupt active car's collisions or scan
# ---------------------------------------------------------------------------
def t10_dnf_no_corruption():
    """
    Force car 1 into DNF via net-displacement window, then verify car 0's
    collision_idx stays clear and its scan is unaffected (DNF car moved off-map).
    """
    reset(2)
    sim_obj = _get_sim()
    engine  = sim_obj._sim

    # Force car 1 to be frozen at its current position (simulate stagnation)
    # by injecting position directly and marking it DNF
    x1 = float(engine.agents[1].state[0])
    y1 = float(engine.agents[1].state[1])

    for _ in range(_DNF_WINDOW_STEPS + 5):
        # Car 1 commanded at 0 — stays still → net disp ≈ 0 → triggers DNF
        apply_action(0, 2.0, 0.0)
        apply_action(1, 0.0, 0.0)
        simulation_step()

    info = get_step_info()
    # Car 1 should be DNF
    assert info["agent_status"][1] == 0, f"car 1 should be DNF, got status={info['agent_status'][1]}"
    assert info["opponents_mask"][1] == False

    # Car 0 must NOT show a vehicle collision with the DNF'd car
    for _ in range(5):
        apply_action(0, 2.0, 0.0)
        simulation_step()
        info2 = get_step_info()
        assert not info2["collisions"]["vehicle"][0], \
            "DNF'd car should not trigger vehicle collision for active car"

    # Car 0's scan should be valid (not corrupted by off-map DNF position)
    obs0 = get_obs(0)
    assert obs0["lidar"].shape == (_LIDAR_RAYS,)
    assert all(obs0["lidar"] >= 0.1)
    assert all(obs0["lidar"] <= 15.0)

# ---------------------------------------------------------------------------
# T-wall — drive into nearest wall: engine iTTC fires, episode does NOT terminate
# ---------------------------------------------------------------------------
def t_wall_hit_physics():
    """Geometry-independent wall hit: orient the car at its min-range LiDAR beam and
    accelerate. collisions.wall[0] must fire and status must stay ACTIVE (no done-on-collision)."""
    reset(1)
    scan = get_obs(0)["lidar"]
    sim_obj = _get_sim()
    st = sim_obj._sim.agents[0].state
    k = int(np.argmin(scan))
    st[4] = float(st[4]) + float(RaceCar.scan_angles[k])   # yaw = state[4]; face nearest wall
    st[3] = 0.0                                            # start from rest; thrust accelerates
    fired = False
    for _ in range(80):
        apply_action(0, _PARAMS["v_max"], 0.0)
        simulation_step()
        info = get_step_info()
        if bool(info["collisions"]["wall"][0]):
            fired = True
            assert info["agent_status"][0] == 1, "wall hit must NOT terminate the episode"
            break
    assert fired, "wall collision never fired while driving into the nearest wall"

# ---------------------------------------------------------------------------
# T15 — progress in [0,1], progress_delta small and plausible
# ---------------------------------------------------------------------------
def t15_progress_bounds():
    reset(1)
    for _ in range(50):
        apply_action(0, 3.0, 0.0)
        simulation_step()
    obs = get_obs(0)
    assert 0.0 <= obs["progress"] < 1.0, f"progress out of [0,1): {obs['progress']}"
    info = get_step_info()
    assert abs(info["progress_delta"][0]) < 0.1, \
        f"progress_delta implausibly large: {info['progress_delta'][0]}"

# ---------------------------------------------------------------------------
# T23 — friction starts at 1.0 immediately
# ---------------------------------------------------------------------------
def t23_friction_initial_value():
    """Friction must be 1.0 (not the engine default 1.0489) right after reset."""
    reset(1)
    info = get_step_info()
    assert info["friction_current"] == _FRICTION_HI, \
        f"Initial friction should be {_FRICTION_HI}, got {info['friction_current']}"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_all([
        ("T05 vehicle collision: no terminate, knockback", t05_vehicle_collision_no_terminate_knockback),
        ("T05b collision with a stopped car",             t05b_collision_with_stopped_car),
        ("T10 DNF no corruption of active cars",          t10_dnf_no_corruption),
        ("T-wall drive into nearest wall (iTTC)",         t_wall_hit_physics),
        ("T15 progress in [0,1], delta plausible",        t15_progress_bounds),
        ("T23 friction starts at 1.0 immediately",       t23_friction_initial_value),
    ])
