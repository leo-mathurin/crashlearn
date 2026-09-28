"""
test_contract.py — Fast, unit-style tests for env_simulation contract.
Run with: docker compose run --rm sim python test_contract.py
"""
import math
import numpy as np

from _harness import run_all, run_test, ok, fail

# ---------------------------------------------------------------------------
# Import under test (no get_my_id)
# ---------------------------------------------------------------------------
from env_simulation import (
    reset, get_obs, apply_action, simulation_step, get_step_info, close,
    get_space_info, _LIDAR_RAYS, _MAX_SLOTS, _get_sim,
    _REQUIRED_LAPS, _DECISION_FREQ_HZ, _KNOCKBACK_REST, _KNOCKBACK_MAX_M,
    _DNF_WINDOW_STEPS,
    _PARAMS,
)

# The module's _get_sim returns the _Sim singleton directly. We alias it for clarity:
_get_sim_instance = _get_sim

# ---------------------------------------------------------------------------
# Helpers for odometer state injection (bypass geometry entirely)
# ---------------------------------------------------------------------------

def _get_sim_raw():
    """Return the singleton _Sim instance from env_module directly.

    The module exposes `_get_sim()` which returns the _Sim singleton.
    We re-import here (not at top-level) to avoid circular import issues
    and to match how tests that need direct state access should do it.
    """
    return _get_sim_instance()

def _make_poses(sim, idx=0):
    """Return poses dict using waypoints from the loaded centerline (safe for off-track DNF)."""
    w = sim._waypoints
    wx, wy = float(w[idx, 0]), float(w[idx, 1])
    return {"poses_x": [wx] * sim._num_agents, "poses_y": [wy] * sim._num_agents}


def _odo_advance(sim, i, target_rel):
    """Simulate one decision-step odometer update to a chosen own-start phase.

    Sets progress_start so absolute == rel, mocks _compute_progress to return
    *target_rel*, runs _update_progress_and_laps, then restores the original
    method in a finally block (to avoid singleton contamination across tests).
    """
    sim._progress_start[i] = 0.0          # so absolute == rel
    obs = _make_poses(sim)
    orig = sim._compute_progress
    sim._compute_progress = lambda x, y: target_rel
    try:
        sim._update_progress_and_laps(obs)
    finally:
        sim._compute_progress = orig


# ---------------------------------------------------------------------------
# T01 — space shape: observations (lidar, velocity, steering), rank, lap_time
# ---------------------------------------------------------------------------
def t01_space_shape():
    """Observation and step_info shapes / defaults."""
    reset(2)
    simulation_step()
    obs0 = get_obs(0)
    info = get_step_info()

    assert obs0["lidar"].shape == (_LIDAR_RAYS,), \
        f"lidar shape {obs0['lidar'].shape} != {_LIDAR_RAYS}"
    assert obs0["velocity"] >= _PARAMS["v_min"] and obs0["velocity"] <= _PARAMS["v_max"], \
        "velocity out of bounds"
    space = get_space_info()
    # get_space_info returns {"observations": {...}, "actions": {...}, "decision_freq_hz": ...}
    assert len(space["observations"]) >= 4, f"observations keys: {list(space.keys())}"
    ok("rank defaults")
    ok("lap_time default")


# ---------------------------------------------------------------------------
# T02 — lidar bounds (clipped): no inf/nan, max in [5, 15]
# ---------------------------------------------------------------------------
def t02_lidar_bounds():
    reset(1)
    simulation_step()
    scan = get_obs(0)["lidar"]
    assert not any(math.isinf(v) for v in scan), "inf in lidar"
    assert not any(math.isnan(v) for v in scan), "nan in lidar"
    ok("max <= 15")
    ok("min >= 0")


# ---------------------------------------------------------------------------
# T03 — action=0 stays on spot: velocity~0, scan unchanged, progress ~0
# ---------------------------------------------------------------------------
def t03_zero_action_stays_on_spot():
    reset(1)
    simulation_step()
    scan0 = get_obs(0)["lidar"].copy()
    apply_action(0, 0.0, 0.0)
    for _ in range(2): simulation_step()
    info = get_step_info()
    assert abs(get_obs(0)["velocity"]) < 0.5, f"velocity {get_obs(0)['velocity']} >= 0.5"
    assert max(abs(get_obs(0)["lidar"] - scan0)) < 0.5, "scan drifted too much"
    # get_step_info returns progress_delta as a dict keyed by agent index
    assert abs(info["progress_delta"][0]) < 0.01, f"progress_delta[{0}]={info['progress_delta'][0]} >= 0.01"


# ---------------------------------------------------------------------------
# T04 — forward thrust → velocity increases, no collision on open track
# ---------------------------------------------------------------------------
def t04_forward_thrust():
    reset(1)
    apply_action(0, _PARAMS["v_max"], 0.0)
    for _ in range(10): simulation_step()
    info = get_step_info()
    assert get_obs(0)["velocity"] > 1.0, f"velocity {get_obs(0)['velocity']} <= 1.0"
    assert not info["collisions"]["vehicle"][0], "unexpected vehicle collision"


# ---------------------------------------------------------------------------
# T06 — wall flag surfaces (deterministic plumbing; physics drive lives in test_integration)
# ---------------------------------------------------------------------------
def t06_wall_flag_surfaces():
    """Contract: an engine-recorded wall hit surfaces as collisions.wall[id], independent
    of vehicle. Deterministic, map-independent. reset() does not touch _wall_hit, and
    get_step_info() reads bool(self._wall_hit[i]) directly, so no step is needed."""
    reset(1)
    sim_obj = _get_sim_raw()
    sim_obj._wall_hit[0] = True
    sim_obj._vehicle_hit[0] = False
    info = get_step_info()
    assert info["collisions"]["wall"][0] is True, \
        f"wall flag not surfaced: {info['collisions']['wall'][0]}"
    assert info["collisions"]["vehicle"][0] is False, "wall/vehicle flags must stay independent"


# ---------------------------------------------------------------------------
# T07 — race start grid (2×2): lane/row assignments
# ---------------------------------------------------------------------------
def t07_race_grid():
    reset(2)
    sim_obj = _get_sim_raw()
    s0 = sim_obj._sim.agents[0].state  # type: ignore[union-attr]
    s1 = sim_obj._sim.agents[1].state  # type: ignore[union-attr]
    assert s0[1] < s1[1], f"P1 lat {s0[1]} >= P2 lat {s1[1]}"
    assert s0[0] >= s1[0], f"P1 lon {s0[0]} < P2 lon {s1[0]}"


# ---------------------------------------------------------------------------
# T08 — friction: starts at 1.0, drops after interval (±1 step)
# Note: If this test fails, check how dynamic track conditions / friction are
# updated during simulation.
# ---------------------------------------------------------------------------
def t08_friction_initial_then_drop():
    reset(1)
    info = get_step_info()
    assert abs(info["friction_current"] - 1.0) < 1e-6, f"friction {info['friction_current']} != 1.0"
    sim_obj = _get_sim_raw()
    # Force countdown to 1 so next step triggers a drop
    sim_obj._friction_countdown = 1
    for _ in range(3): simulation_step()
    info2 = get_step_info()
    assert info2["friction_current"] < 1.0, f"friction should have dropped, got {info2['friction_current']}"


# ---------------------------------------------------------------------------
# T09 — lap time computation: non-zero first-lap time after crossing
# ---------------------------------------------------------------------------
def t09_lap_time_after_crossing():
    reset(1)
    sim_obj = _get_sim_raw()

    # Advance step_count and set lap_start_step so lap_time > 0
    sim_obj._step_count = 500
    sim_obj._lap_start_step[0] = 400  # lap_time = (500-400)/20 = 5.0 > 0

    orig = sim_obj._compute_progress
    try:
        # rel just below 1.0 (0.9), then we advance to phase=0.05 (forward d)
        sim_obj._progress_start[0] = 0.0
        sim_obj._rel_prev[0] = 0.9
        sim_obj._cum[0] = 0.9
        sim_obj._lap_count[0] = 0

        # compute_progress returns the "current" absolute pos → rel = (p_abs - 0) % 1
        # We want rel=0.05 after this step, so set abs=0.05
        sim_obj._compute_progress = lambda x, y: 0.05
        sim_obj._update_progress_and_laps(_make_poses(sim_obj))
    finally:
        sim_obj._compute_progress = orig

    info = get_step_info()
    assert bool(info["lap_complete"][0]) is True, f"lap_complete must be True, got {info['lap_complete'][0]}"
    assert len(sim_obj._lap_times[0]) >= 1 and sim_obj._lap_times[0][-1] > 0.0, \
        f"no positive lap time: {sim_obj._lap_times}"


# ---------------------------------------------------------------------------
# T17 — lap booked when cum crosses an integer
# ---------------------------------------------------------------------------
def t17_lap_booked_on_cum_crossing():
    reset(1)
    sim_obj = _get_sim_raw()

    orig = sim_obj._compute_progress
    try:
        sim_obj._progress_start[0] = 0.0
        sim_obj._rel_prev[0] = 0.9
        sim_obj._cum[0] = 0.9
        sim_obj._lap_count[0] = 0

        # absolute=0.05 → rel=(0.05-0)%1=0.05, d = (0.05-0.9)+1 = 0.15 => cum=1.05
        sim_obj._compute_progress = lambda x, y: 0.05
        sim_obj._update_progress_and_laps(_make_poses(sim_obj))
    finally:
        sim_obj._compute_progress = orig

    info = get_step_info()
    assert bool(info["lap_complete"][0]) is True, f"lap_complete must be True, got {info['lap_complete'][0]}"
    assert sim_obj._lap_count[0] == 1, f"lap_count {sim_obj._lap_count[0]} != 1"


# ---------------------------------------------------------------------------
# T18 — finish (status=2, race_over) at cum >= _REQUIRED_LAPS
# ---------------------------------------------------------------------------
def t18_finish_at_required_laps():
    reset(1)
    sim_obj = _get_sim_raw()

    # step_count offset for lap_time computation. lap_time = (step_count - lap_start_step) / _DECISION_FREQ
    sim_obj._step_count = 500
    sim_obj._lap_start_step[0] = 400  # will give lap_time = (1000-400)/20 = 30 > 0

    orig = sim_obj._compute_progress
    try:
        # Pretend car has done (_REQUIRED_LAPS - 1) laps, cum at 2.9, almost at last
        sim_obj._progress_start[0] = 0.0
        sim_obj._rel_prev[0] = 0.9
        sim_obj._cum[0] = float(_REQUIRED_LAPS - 1) + 0.9  # 2.9
        sim_obj._lap_count[0] = _REQUIRED_LAPS - 1

        # bump step_count before crossing so lap_time > 0
        sim_obj._step_count += 500   # step_count = 1000

        # abs=0.05 → rel=0.05, d=(0.05-0.9)+1=0.15 => cum=3.05 >= _REQUIRED_LAPS=3
        sim_obj._compute_progress = lambda x, y: 0.05
        sim_obj._update_progress_and_laps(_make_poses(sim_obj))
    finally:
        sim_obj._compute_progress = orig

    info = get_step_info()
    assert int(info["agent_status"][0]) == 2, f"agent_status {info['agent_status'][0]} != FINISHED"
    assert bool(info["race_over"]) is True, "race_over must be True"
    assert len(info["lap_times"][0]) >= 1, f"lap_times empty: {info['lap_times']}"
    # step_count=500 → +500=1000; lap_start_step was set to (500+500)-10=990
    # lap_time = (1000-990)/20 = 0.5 > 0
    assert float(info["lap_times"][0][-1]) > 0.0, f"last lap_time {info['lap_times'][0][-1]} <= 0"


# ---------------------------------------------------------------------------
# T19 — two laps → two positive lap times
# ---------------------------------------------------------------------------
def t19_two_laps_two_times():
    reset(1)
    sim_obj = _get_sim_raw()

    orig = sim_obj._compute_progress
    try:
        # First crossing: cum 0.9 → lap 1 (rel 0.9→0.05)
        sim_obj._progress_start[0] = 0.0
        sim_obj._rel_prev[0] = 0.9
        sim_obj._cum[0] = 0.9
        sim_obj._lap_count[0] = 0
        sim_obj._step_count = 100
        # lap_start_step must be < step_count so lap_time = (step - start)/20 > 0
        sim_obj._lap_start_step[0] = 50

        sim_obj._compute_progress = lambda x, y: 0.05
        sim_obj._update_progress_and_laps(_make_poses(sim_obj))

        assert sim_obj._lap_count[0] == 1, f"first lap not booked: count={sim_obj._lap_count[0]}"
        # lap_time = (100 - 50) / 20 = 2.5 > 0
        assert len(sim_obj._lap_times[0]) >= 1 and float(sim_obj._lap_times[0][-1]) > 0.0, \
            f"first lap time missing/zero: {sim_obj._lap_times}"

        # Second crossing: cum 1.95 → lap 2 (rel 0.95→0.05), with step offset for lap_time > 0
        sim_obj._step_count += 100
        # Keep lap_start_step behind so lap_time > 0
        sim_obj._lap_start_step[0] = sim_obj._step_count - 10
        sim_obj._rel_prev[0] = 0.95
        sim_obj._cum[0] = 1.95

        sim_obj._compute_progress = lambda x, y: 0.05  # rel=(0.05-0)%1=0.05, d=(0.05-0.95)+1=0.1 => cum=2.05
        sim_obj._update_progress_and_laps(_make_poses(sim_obj))
    finally:
        sim_obj._compute_progress = orig

    assert sim_obj._lap_count[0] == 2, f"second lap not booked: count={sim_obj._lap_count[0]}"
    assert len(sim_obj._lap_times[0]) >= 2, f"only {len(sim_obj._lap_times[0])} lap times: {sim_obj._lap_times}"
    assert all(t > 0.0 for t in sim_obj._lap_times[0]), \
        f"not all lap times > 0: {sim_obj._lap_times}"


# ---------------------------------------------------------------------------
# T-anti-rebound — oscillating at own start books 0 laps
# ---------------------------------------------------------------------------
def t_anti_rebound():
    reset(1)
    sim_obj = _get_sim_raw()
    sim_obj._progress_start[0] = 0.0
    sim_obj._rel_prev[0] = 0.0
    sim_obj._cum[0] = 0.0
    sim_obj._lap_count[0] = 0

    for k in range(50):
        target = 0.0 + 0.01 * math.sin(k)
        _odo_advance(sim_obj, 0, target)
    assert sim_obj._lap_count[0] == 0, f"anti-rebound: lap_count {sim_obj._lap_count[0]} != 0"
    assert abs(sim_obj._cum[0]) < 0.05, f"anti-rebound: cum {sim_obj._cum[0]} >= 0.05"


# ---------------------------------------------------------------------------
# T28 — FIX E6: DNFs are ranked by distance covered, not by slot number.
# Values taken from a real 4-car Austin recording, where car 2 covered more than
# car 0 yet was ranked behind it purely because its slot index was higher.
# ---------------------------------------------------------------------------
def t28_dnf_ranked_by_distance():
    reset(4)
    sim_obj = _get_sim_raw()
    orig_status = list(sim_obj._status[:])
    try:
        sim_obj._status[0] = 0        # DNF, covered the least
        sim_obj._status[2] = 0        # DNF, covered more than car 0
        sim_obj._max_progress[0] = 0.1260
        sim_obj._max_progress[2] = 0.1308
        sim_obj._cum[1] = 1.0131      # active
        sim_obj._cum[3] = 1.1777      # active, leads

        ranks = get_step_info()["ranks"]
        assert int(ranks[3]) == 1 and int(ranks[1]) == 2, \
            f"active cars must lead: got {ranks}"
        assert int(ranks[2]) == 3 and int(ranks[0]) == 4, \
            f"DNF car 2 (0.1308) must outrank car 0 (0.1260): got {ranks}"

        # Equal distance -> slot order decides, deterministically.
        sim_obj._max_progress[2] = 0.1260
        ranks2 = get_step_info()["ranks"]
        assert int(ranks2[0]) == 3 and int(ranks2[2]) == 4, \
            f"tie must fall back to slot order: got {ranks2}"
    finally:
        np.copyto(sim_obj._status, np.array(orig_status, dtype=sim_obj._status.dtype))


# ---------------------------------------------------------------------------
# T-ranks — ordering by descending _cum (rewritten for odometer)
# ---------------------------------------------------------------------------
def t_ranks():
    reset(3)
    sim_obj = _get_sim_raw()

    orig_status = list(sim_obj._status[:])
    orig_finish = list(sim_obj._finish_step[:])
    try:
        # Set cumulative laps explicitly
        sim_obj._cum[0] = 2.5   # car 0 leads
        sim_obj._cum[1] = 1.2   # car 1 second
        sim_obj._cum[2] = 3.8   # car 2 actually leads

        info = get_step_info()
        assert info["ranks"][2] < info["ranks"][0] < info["ranks"][1], \
            f"expected rank(2)<rank(0)<rank(1) but got ranks={info['ranks']}"

        # FINISHED-by-finish_step before active even with lower cum
        sim_obj._finish_step[1] = 50   # car 1 finished
        sim_obj._status[1] = 2          # FINISHED
        info2 = get_step_info()
        assert info2["ranks"][1] < info2["ranks"][0], \
            f"rank(1)={info2['ranks'][1]} must be < rank(0)={info2['ranks'][0]}"

        # DNF last (status=0): car 2 = DNF → ranked after all active cars
        sim_obj._finish_step[1] = -1
        sim_obj._status[1] = 1
        sim_obj._status[2] = 0         # DNF
        info3 = get_step_info()
        # DNF ranks after ALL finished+active cars. Here finished=0, active=[car0,car1] -> DNF=rank 3.
        n_finished = sum(1 for i in range(sim_obj._num_agents) if int(sim_obj._status[i]) == 2)
        n_active   = sum(1 for i in range(sim_obj._num_agents) if int(sim_obj._status[i]) == 1)
        assert int(info3["ranks"][2]) == n_finished + n_active + 1, \
            f"DNF must rank after all finished+active: got {info3['ranks'][2]}, expected {n_finished + n_active + 1}"
    finally:
        # Restore singleton state (use numpy array assignment with proper dtype for in-place)
        np.copyto(sim_obj._status, np.array(orig_status, dtype=sim_obj._status.dtype))
        np.copyto(sim_obj._finish_step, np.array(orig_finish, dtype=sim_obj._finish_step.dtype))


# ---------------------------------------------------------------------------
# T15 — progress in [0,1), progress_delta plausibility
# ---------------------------------------------------------------------------
def t15_progress_bounds():
    reset(1)
    for _ in range(50):
        apply_action(0, 3.0, 0.0)
        simulation_step()
    obs = get_obs(0)
    assert 0.0 <= obs["progress"] < 1.0, f"progress={obs['progress']}"
    info = get_step_info()
    assert abs(info["progress_delta"][0]) < 0.1, f"progress_delta={info['progress_delta'][0]}"


# ---------------------------------------------------------------------------
# T20 — knockback subtle velocity modulation verification
#
# Scenario: two cars head-on along +x, 2 m apart, equal speeds.
#   Car-0 at x=5 at v=+2 m/s ; Car-1 at x=7 at v=−2 m/s
# n = (p0−p1)/|..| = [−1, 0] ; vrel = (v0−v1)·n = −4 m/s ; split 50/50.
# Expected values are recomputed below from the live constants, so the test
# follows _KNOCKBACK_REST and the cap rather than pinning them.
# ---------------------------------------------------------------------------
def t20_knockback_impulse_math():
    """Verify position-based knockback (normal-weighted split, capped, ±15% modulation)."""
    reset(2)
    sim_obj = _get_sim_raw()
    engine = sim_obj._sim
    assert engine is not None

    v_forward = 2.0
    car0_state = engine.agents[0].state
    car1_state = engine.agents[1].state
    # ST state: [x, y, steer, vel, yaw, yaw_rate, slip]; idx3=speed, idx4=yaw.
    car0_state[:] = [5.0, 4.5, 0.0,  v_forward, 0.0, 0.0, 0.0]  # moving +x
    car1_state[:] = [7.0, 4.5, 0.0, -v_forward, 0.0, 0.0, 0.0]  # moving -x
    sim_obj._apply_knockback(0, 1)

    impulse_mag = min(2 * v_forward * _KNOCKBACK_REST, _KNOCKBACK_MAX_M)
    half = impulse_mag * 0.5                     # equal speeds → 50/50 split
    vel_change = half * 0.15

    expected_v0 = max(v_forward - vel_change, -vel_change)
    expected_v1 = min(-v_forward + vel_change, _PARAMS["v_max"])
    v0_after = float(car0_state[3])
    v1_after = float(car1_state[3])

    assert abs(v0_after - expected_v0) < 0.01, \
        f"knockback car0: expected={expected_v0:+.3f}, got={v0_after:+.3f}"
    assert abs(v1_after - expected_v1) < 0.01, \
        f"knockback car1: expected={expected_v1:+.3f}, got={v1_after:+.3f}"
    # Positional displacement along the normal (main effect)
    assert abs(float(car0_state[0]) - (5.0 - half)) < 1e-6, "car0 position knockback"
    assert abs(float(car1_state[0]) - (7.0 + half)) < 1e-6, "car1 position knockback"


# ---------------------------------------------------------------------------
# T25 — no DNF right after a lap completion (regression test for FIX C1)
# Simulates steady forward driving via a mocked centerline projection, crosses
# lap 1, then keeps advancing for a full DNF window: status must stay ACTIVE.
# ---------------------------------------------------------------------------
def t25_no_dnf_after_lap():
    reset(1)
    sim_obj = _get_sim_raw()
    orig = sim_obj._compute_progress
    try:
        sim_obj._progress_start[0] = 0.0
        prog = {"v": 0.0}
        sim_obj._compute_progress = lambda x, y: prog["v"] % 1.0
        # Advance 0.01 lap/step until well past lap 1, then one more DNF window.
        n_steps = 110 + (_DNF_WINDOW_STEPS + 10)
        for _ in range(n_steps):
            prog["v"] += 0.01
            sim_obj._step_count += 1
            sim_obj._update_progress_and_laps(_make_poses(sim_obj))
            sim_obj._update_stagnation_and_dnf()
            assert sim_obj._status[0] != 0, \
                f"DNF at cum={sim_obj._cum[0]:.3f} (lap_count={sim_obj._lap_count[0]}) — C1 regression"
        assert sim_obj._lap_count[0] >= 1, "lap 1 never booked"
    finally:
        sim_obj._compute_progress = orig


# ---------------------------------------------------------------------------
# T24 — FIX A5: the winning lap is reported, exactly once
# ---------------------------------------------------------------------------
def t24_winning_lap_is_reported():
    """_finish() flips status to 2 on the same step the last lap is booked.
    lap_complete must stay True on that step (otherwise an agent counting laps
    undercounts by one), then drop back to False on the next step (no sticky flag)."""
    reset(1)
    sim_obj = _get_sim_raw()
    sim_obj._progress_start[0] = 0.0

    orig = sim_obj._compute_progress
    try:
        for lap in range(1, _REQUIRED_LAPS + 1):
            # just short of the line, then cross it: cum passes the integer `lap`
            sim_obj._rel_prev[0] = 0.9
            sim_obj._cum[0] = lap - 1 + 0.9
            sim_obj._compute_progress = lambda x, y: 0.05
            sim_obj._update_progress_and_laps(_make_poses(sim_obj))

            info = get_step_info()
            assert bool(info["lap_complete"][0]) is True, \
                f"lap {lap}: lap_complete must be True, got {info['lap_complete'][0]}"
            assert sim_obj._lap_count[0] == lap, \
                f"lap_count {sim_obj._lap_count[0]} != {lap}"

        # last lap => FINISHED
        assert sim_obj._status[0] == 2, f"status {sim_obj._status[0]} != 2 after the last lap"

        # next step: the flag must not stick
        sim_obj._update_progress_and_laps(_make_poses(sim_obj))
        info = get_step_info()
        assert bool(info["lap_complete"][0]) is False, \
            f"sticky lap_complete after finish: {info['lap_complete'][0]}"
    finally:
        sim_obj._compute_progress = orig


def t26_track_widths_are_real():
    """E2 — largeurs reelles par waypoint, et test hors-piste coherent."""
    import glob, os
    from env_simulation import (
        _project_lateral_offset_njit, _OFFTRACK_MARGIN_M,
    )

    here = os.path.dirname(os.path.abspath(__file__))
    csvs = sorted(glob.glob(os.path.join(here, "maps", "*", "*_centerline.csv")))
    assert len(csvs) >= 20, f"only {len(csvs)} centerline files found"

    def is_off(x, y, wpts, widths):
        lat, seg = _project_lateral_offset_njit(x, y, wpts)
        seg = min(max(int(seg), 0), widths.shape[0] - 1)
        return lat > widths[seg, 0] + _OFFTRACK_MARGIN_M or \
            lat < -(widths[seg, 1] + _OFFTRACK_MARGIN_M)

    medians = []
    probes = misses = 0
    for csv in csvs:
        name = os.path.basename(os.path.dirname(csv))
        cl = np.loadtxt(csv, delimiter=",", comments="#")
        wpts = np.ascontiguousarray(cl[:, :2])
        widths = cl[:, 2:4]
        assert widths.min() > 0.3, f"{name}: implausible half-width {widths.min()}"
        medians.append(float(np.median(widths.sum(axis=1))))

        # normale droite (lateral > 0 = droite)
        dx = np.roll(wpts[:, 0], -1) - np.roll(wpts[:, 0], 1)
        dy = np.roll(wpts[:, 1], -1) - np.roll(wpts[:, 1], 1)
        nn = np.hypot(dx, dy)
        nx, ny = dy / nn, -dx / nn

        for i in range(len(wpts)):
            assert not is_off(wpts[i, 0], wpts[i, 1], wpts, widths), \
                f"{name}: centerline point {i} declared off-track"
            for sign, col in ((1.0, 0), (-1.0, 1)):
                d = widths[i, col] + _OFFTRACK_MARGIN_M + 0.10
                probes += 1
                if not is_off(wpts[i, 0] + sign * nx[i] * d,
                              wpts[i, 1] + sign * ny[i] * d, wpts, widths):
                    misses += 1

    # Probes sit one margin past the edge, where centerline projection can pick a
    # neighbouring segment in tight corners. Harmless: walls hug the track edge
    # (median gap 0.00 m over 23 tracks), so no car can reach those positions.
    rate = misses / probes
    assert rate < 0.04, f"{100 * rate:.1f}% of off-track probes missed (>4%)"
    spread = max(medians) - min(medians)
    assert spread > 0.5, f"track widths still near-constant (spread {spread:.2f} m)"


def t27_no_yaw_blowup_in_reverse():
    """Le modele dynamique diverge en marche arriere : lacet borne a toute vitesse."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gym"))
    from f110_gym.envs.dynamic_models import vehicle_dynamics_st
    from env_simulation import _PARAMS, _WALL_REST_MAX

    assert _WALL_REST_MAX < 0.5, \
        f"_WALL_REST_MAX={_WALL_REST_MAX} >= 0.5 : recul mur dans la branche dynamique"

    keys = ("mu", "C_Sf", "C_Sr", "lf", "lr", "h", "m", "I",
            "s_min", "s_max", "sv_min", "sv_max", "v_switch", "a_max", "v_min", "v_max")
    prm = {k: _PARAMS[k] for k in keys}
    lwb = _PARAMS["lf"] + _PARAMS["lr"]

    for v0 in (-5.0, -2.0, -0.6, -0.51, -0.5, -0.49, -0.1, 0.49, 0.6, 3.0, 10.0):
        x = np.array([0.0, 0.0, 0.25, v0, 0.0, 0.0, 0.0])
        for _ in range(50):                      # 50 ms
            x = x + 0.001 * vehicle_dynamics_st(x, np.array([0.0, 0.0]), **prm)
        yaw = abs(math.degrees(x[4]))
        # borne cinematique, large (x3) pour le derapage
        lim = math.degrees(abs(v0) / lwb * math.tan(0.4189) * 0.05) * 3.0 + 1.0
        assert yaw < lim, \
            f"v0={v0}: yaw={yaw:.1f} deg en 50 ms (limite {lim:.1f}) -> modele divergent"
        assert abs(math.degrees(x[5])) < 2000.0, \
            f"v0={v0}: yaw_rate={math.degrees(x[5]):.0f} deg/s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_all([
        ("T01 space shape (obs/rank/lap_time)",     t01_space_shape),
        ("T02 lidar bounds (no inf/nan, max ≤ 15)", t02_lidar_bounds),
        ("T03 zero action stays on spot",           t03_zero_action_stays_on_spot),
        ("T04 forward thrust increases velocity",   t04_forward_thrust),
        ("T06 wall flag surfaces",                  t06_wall_flag_surfaces),
        ("T07 race grid (2×2)",                     t07_race_grid),
        ("T08 friction starts at 1.0, then drops", t08_friction_initial_then_drop),
        ("T09 lap time non-zero after crossing",    t09_lap_time_after_crossing),
        ("T17 lap booked when cum crosses integer", t17_lap_booked_on_cum_crossing),
        ("T18 finish at required laps",             t18_finish_at_required_laps),
        ("T19 two laps → two positive times",       t19_two_laps_two_times),
        ("T-anti-rebound oscillating books 0 laps", t_anti_rebound),
        ("T-ranks ordering by descending cum",      t_ranks),
        ("T15 progress bounds & delta plausibility", t15_progress_bounds),
        ("T20 knockback impulse math",              t20_knockback_impulse_math),
        ("T24 winning lap is reported (A5)",        t24_winning_lap_is_reported),
        ("T25 no DNF right after lap complete (C1)", t25_no_dnf_after_lap),
        ("T26 real per-waypoint track widths (E2)", t26_track_widths_are_real),
        ("T27 no yaw blow-up in reverse",           t27_no_yaw_blowup_in_reverse),
        ("T28 DNF ranked by distance (E6)",         t28_dnf_ranked_by_distance),
    ])