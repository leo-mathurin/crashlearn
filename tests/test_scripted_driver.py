"""Scripted diagnostic driver: determinism, bounded finite actions, isolation, CLI guards."""

import json
import math
import subprocess
import sys

import numpy as np
import pytest
from agent_loader import create_dummy_info, create_dummy_obs
from conftest import ROOT
from test_submission_contract import check_action

from crashlearn.scripted_driver import DriverConfig, ScriptedDriver

SCRIPT = ROOT / "scripts" / "control_race.py"


def obs(lidar, *, agent_id=0, velocity=2.0, progress=0.5, opponents=None):
    return {
        "agent_id": agent_id,
        "lidar": np.asarray(lidar, dtype=np.float32),
        "velocity": velocity,
        "steering": 0.0,
        "progress": progress,
        "lap_count": 0,
        "rank": 1,
        "opponents": opponents or {},
    }


def info(step, *, wall=False, agent_id=0):
    walls = {i: wall and i == agent_id for i in range(4)}
    return {
        "step_count": step,
        "collisions": {"wall": walls, "vehicle": dict.fromkeys(walls, False)},
    }


def episode(seed=0, steps=60):
    """Synthetic but varied inputs, including wall contacts and a stuck phase."""
    rng = np.random.default_rng(seed)
    return [
        (
            obs(
                rng.uniform(0.1, 15.0, 100),
                velocity=0.0 if 20 <= k < 35 else 3.0,
                progress=(0.3 + 0.001 * k) % 1.0,
            ),
            info(k, wall=k in (10, 45)),
        )
        for k in range(steps)
    ]


def run(driver, inputs):
    return [driver.predict(o, i) for o, i in inputs]


def test_same_inputs_give_same_actions():
    inputs = episode()
    driver = ScriptedDriver()
    first = run(driver, inputs)
    assert first == run(ScriptedDriver(), inputs)
    assert first == run(driver, inputs)  # step_count == 0 resets the episode state
    assert any(speed < 0 for speed, _ in first), "episode should exercise the reverse path"


# --- action contract: every config variant (league opponents) and every degraded input ---

NAN, INF = float("nan"), float("inf")
CONFIGS = {
    "default": DriverConfig(),
    "reverse_fast": DriverConfig(reverse_speed=-5.0),
    "speeds_negative": DriverConfig(min_speed=-10.0, max_speed=-10.0),
    "speeds_zero": DriverConfig(min_speed=0.0, max_speed=0.0, speed_gain=0.0),
    "speeds_high": DriverConfig(min_speed=20.0, max_speed=20.0, speed_gain=10.0),
    "gains_zero": DriverConfig(steer_gain=0.0, steer_slowdown=0.0),
    "gains_high": DriverConfig(steer_gain=10.0, steer_slowdown=10.0),
    "steps_zero": DriverConfig(stuck_steps=0, reverse_steps=0, wrong_way_steps=0),
    "steps_one": DriverConfig(stuck_steps=1, reverse_steps=1, wrong_way_steps=1),
    "steps_many": DriverConfig(stuck_steps=100, reverse_steps=100, wrong_way_steps=100),
    "windows_zero": DriverConfig(fov_deg=0.0, front_cone_deg=0.0),
    "windows_one_degree": DriverConfig(fov_deg=1.0, front_cone_deg=1.0),
    "windows_full": DriverConfig(fov_deg=180.0, front_cone_deg=180.0),
    "thresholds_zero": DriverConfig(
        disparity_threshold=0.0,
        car_half_width=0.0,
        margin=0.0,
        target_tolerance=0.0,
        front_margin=0.0,
        stuck_speed=0.0,
        no_reverse_below_progress=0.0,
    ),
    "thresholds_high": DriverConfig(
        disparity_threshold=100.0,
        car_half_width=100.0,
        margin=100.0,
        target_tolerance=100.0,
        front_margin=100.0,
        stuck_speed=100.0,
        no_reverse_below_progress=100.0,
    ),
    "nan_values": DriverConfig(max_speed=NAN, steer_gain=NAN),
    "inf_values": DriverConfig(max_speed=INF, steer_gain=INF),
}
LIDAR = np.full(100, 5.0)
INPUTS = {
    "lidar_min": {"lidar": np.full(100, 0.1)},
    "lidar_max": {"lidar": np.full(100, 15.0)},
    "lidar_nan": {"lidar": np.full(100, NAN)},
    "lidar_inf": {"lidar": np.where(np.arange(100) % 2, INF, -INF)},
    "lidar_none": {"lidar": None},
    "lidar_2d": {"lidar": np.full((1, 100), 5.0)},
    "lidar_empty": {"lidar": []},
    "lidar_short": {"lidar": np.full(50, 3.0)},
    "opponent_status_text": {"lidar": LIDAR, "opponents": {1: {"status": "1", "x_rel": 1.0}}},
    "opponent_position_nan": {
        "lidar": LIDAR,
        "opponents": {1: {"status": 1, "x_rel": NAN, "y_rel": NAN}},
    },
    "opponent_ahead": {"lidar": LIDAR, "opponents": {1: {"status": 1, "x_rel": 0.5, "y_rel": 0}}},
    "loader_dummy": create_dummy_obs(),
}


def contract_sequence(base):
    """30 decisions: forward, wrong way, stuck, then wall contacts next to the line."""
    for k in range(30):
        if k < 10:
            velocity, progress = 3.0, 0.5 + 0.001 * k
        elif k < 20:
            velocity, progress = 3.0, 0.52 - 0.002 * (k - 10)
        elif k < 25:
            velocity, progress = 0.0, 0.5
        else:
            velocity, progress = NAN if k == 26 else 0.0, 0.005
        yield {**base, "velocity": velocity, "progress": progress}, info(k, wall=k in (5, 22, 27))


@pytest.mark.parametrize("base", INPUTS.values(), ids=INPUTS.keys())
@pytest.mark.parametrize("config", CONFIGS.values(), ids=CONFIGS.keys())
def test_action_contract(config, base):
    driver = ScriptedDriver(config)
    for o, i in contract_sequence(base):
        check_action(driver.predict(o, i))
    check_action(driver.predict({}, {}))
    check_action(driver.predict(create_dummy_obs(), create_dummy_info()))


def test_step_count_reset():
    config = DriverConfig()
    stuck = obs(LIDAR, velocity=0.0)

    missing = ScriptedDriver(config)  # no step_count: state must persist, recovery must fire
    speeds = [missing.predict(stuck, {})[0] for _ in range(config.stuck_steps)]
    assert speeds[-1] == config.reverse_speed

    kept = ScriptedDriver(config)
    kept.predict(obs(LIDAR), info(0))
    assert kept.predict(obs(LIDAR), info(1, wall=True))[0] == config.reverse_speed
    assert kept.predict(obs(LIDAR), info(2))[0] == config.reverse_speed  # still reversing

    kept.predict(obs(LIDAR), info(1, wall=True))
    assert kept.predict(obs(LIDAR), info(0))[0] > 0  # step 0 = new episode, reverse dropped


def test_instances_share_no_state():
    a, b = episode(seed=1), episode(seed=2)
    alone = (run(ScriptedDriver(), a), run(ScriptedDriver(), b))
    da, db = ScriptedDriver(), ScriptedDriver()
    together = ([], [])
    for (oa, ia), (ob, ib) in zip(a, b, strict=True):
        together[0].append(da.predict(oa, ia))
        together[1].append(db.predict(ob, ib))
    assert together == alone


def test_wall_contact_reverses_slowly_except_near_the_line():
    config = DriverConfig()
    assert -0.5 < config.reverse_speed < 0  # faster steered reversing diverges in the engine
    driver = ScriptedDriver(config)
    driver.predict(obs(np.full(100, 5.0)), info(0))
    actions = [driver.predict(obs(np.full(100, 5.0)), info(1, wall=True))]
    actions += [driver.predict(obs(np.full(100, 5.0)), info(k)) for k in range(2, 30)]
    speeds = [speed for speed, _ in actions]
    assert speeds[: config.reverse_steps] == [config.reverse_speed] * config.reverse_steps
    assert speeds[config.reverse_steps] > 0

    near_line = ScriptedDriver(config)
    near_line.predict(obs(np.full(100, 5.0), progress=0.0), info(0))
    speed, _ = near_line.predict(obs(np.full(100, 5.0), progress=0.01), info(1, wall=True))
    assert speed == config.min_speed


def test_opponent_ahead_is_avoided():
    open_track = obs(np.full(100, 15.0))
    ahead = {1: {"x_rel": 1.0, "y_rel": 0.0, "status": 1}}
    blocked = obs(np.full(100, 15.0), opponents=ahead)
    speed_free, steer_free = ScriptedDriver().predict(open_track, info(1))
    speed_blocked, steer_blocked = ScriptedDriver().predict(blocked, info(1))
    assert abs(steer_free) < 0.05  # nearest rays to straight ahead are at +-1.8 deg
    assert abs(steer_blocked) > 0.1
    assert speed_blocked < speed_free
    inactive = obs(np.full(100, 15.0), opponents={1: {**ahead[1], "status": 0}})
    assert ScriptedDriver().predict(inactive, info(1)) == (speed_free, steer_free)


@pytest.mark.parametrize(
    "args",
    [
        ["--map", "Spa"],  # sealed test split
        ["--map", "Montreal"],  # validation split
        ["--map", "Mexico City"],  # alias, not canonical
        ["--map", "Monaco"],
        ["--laps", "4"],  # the simulator finishes a car at 3 laps
        ["--cars", "5"],
        ["--max-time", "0"],
    ],
)
def test_control_race_rejects_invalid_arguments(args, tmp_path):
    out = tmp_path / "out.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--output", str(out)], capture_output=True, text=True
    )
    assert result.returncode == 2, result.stderr
    assert not out.exists()


@pytest.mark.slow
def test_control_race_is_reproducible_with_provenance(tmp_path):
    runs = []
    for name in ("a", "b"):
        out = tmp_path / f"{name}.json"
        cmd = [sys.executable, str(SCRIPT), "--map", "IMS", "--cars", "2", "--max-time", "15"]
        subprocess.run([*cmd, "--output", str(out)], check=True, capture_output=True)
        runs.append(json.loads(out.read_text()))
    for record in runs[0]:
        record.pop("created_utc")
        assert record["commit"] and record["sim_sha256"] and record["driver_config"]
        assert (record["map"], record["seed"], record["laps"]) == ("IMS", 0, 3)
        assert record["status"] == "TIMEOUT"
        assert record["decisions"] == 300
        assert 0 < record["progress_laps"] < 1
        assert math.isfinite(record["distance_m"]) and record["distance_m"] > 0
    for record in runs[1]:
        record.pop("created_utc")
    assert runs[0] == runs[1]
    assert [r["car"] for r in runs[0]] == [0, 1]
