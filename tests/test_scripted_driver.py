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


@pytest.mark.parametrize(
    "lidar",
    [
        np.full(100, 0.1),
        np.full(100, 15.0),
        np.full(100, np.nan),
        np.where(np.arange(100) % 2, np.inf, -np.inf),
        np.full(50, 3.0),
        [],
    ],
    ids=["all_min", "all_max", "nan", "inf", "short", "empty"],
)
@pytest.mark.parametrize("velocity", [0.0, 3.0, float("nan")])
def test_actions_bounded_and_finite(lidar, velocity):
    driver = ScriptedDriver()
    for k in range(40):  # walls on every other step: forward, reverse and stuck paths
        o = obs(lidar, velocity=velocity, progress=float("nan") if k == 5 else 0.5)
        check_action(driver.predict(o, info(k, wall=k % 2 == 1)))


def test_official_dummy_inputs():
    check_action(ScriptedDriver().predict(create_dummy_obs(), create_dummy_info()))


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
