"""CrashLearnEnv episodes (E-14): units, episode ends, laps, overrides, singleton.

Driven on the example circuit: its start straight hits a wall after ~10 steps at full speed.
"""

import env_simulation as es
import numpy as np
import pytest
from conftest import pure_pursuit

from crashlearn.env import (
    OBS_SIZE,
    STATUS_ACTIVE,
    STATUS_DNF,
    CrashLearnEnv,
    action_to_controls,
    controls_to_action,
    encode_observation,
)

IDLE = controls_to_action(0.0, 0.0)
FULL_THROTTLE = controls_to_action(10.0, 0.0)


@pytest.fixture(scope="module", autouse=True)
def _restore_shared_sim_state():
    """Closing an env destroys the singleton: give later modules the 2-car example state
    that `conftest.sim` hands out."""
    yield
    es.set_map("example")
    es.reset(2)
    es.simulation_step()


@pytest.fixture
def make_env():
    envs = []

    def make(**kwargs):
        env = CrashLearnEnv(**{"maps": "example", **kwargs})
        envs.append(env)
        return env

    yield make
    for env in envs:
        env.close()


def drive(env, policy, steps):
    """Run `policy(info) -> action` until the episode ends; return the per-step transitions."""
    _, info = env.reset(seed=0)
    out = []
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(policy(info))
        out.append((obs, reward, terminated, truncated, info))
        if terminated or truncated:
            break
    return out


def pursuit(speed):
    return lambda info: controls_to_action(*pure_pursuit(es._get_sim(), 0, speed=speed))


# --- Units ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "controls"),
    [
        ((-1.0, -1.0), (-2.0, -0.4189)),
        ((1.0, 1.0), (10.0, 0.4189)),
        ((-2 / 3, 0.0), (0.0, 0.0)),
        ((5.0, -5.0), (10.0, -0.4189)),  # out-of-box actions are clipped
    ],
)
def test_action_maps_to_simulator_units(action, controls):
    assert action_to_controls(np.array(action)) == pytest.approx(controls)


def test_controls_to_action_inverts_the_mapping():
    for speed, steer in [(-2.0, 0.1), (3.0, -0.4), (10.0, 0.0)]:
        assert action_to_controls(controls_to_action(speed, steer)) == pytest.approx(
            (speed, steer), abs=1e-6
        )


def test_encoded_observation_layout_and_units():
    raw = {
        "lidar": np.full(100, 7.5, dtype=np.float32),
        "velocity": 5.0,
        "steering": -0.4189,
        "progress": 0.25,
        "lap_count": 3,
    }
    vec = encode_observation(raw)
    assert vec.shape == (OBS_SIZE,) and vec.dtype == np.float32
    np.testing.assert_allclose(vec[:100], 0.5)
    np.testing.assert_allclose(vec[100:], [0.5, -1.0, 1.0, 0.0, 1.0], atol=1e-6)


def test_commanded_speed_is_reached_in_metres_per_second(make_env):
    env = make_env()
    steps = drive(env, pursuit(2.0), 40)
    assert steps[-1][4]["raw_obs"]["velocity"] == pytest.approx(2.0, abs=0.05)
    assert steps[-1][0][100] == pytest.approx(0.2, abs=0.005)


# --- Episode ends ---------------------------------------------------------


def test_idle_car_terminates_as_dnf_after_the_stagnation_window(make_env):
    env = make_env()
    steps = drive(env, lambda info: IDLE, 400)
    assert len(steps) == 201
    *_, terminated, truncated, info = steps[-1]
    assert terminated and not truncated
    assert info["status"] == STATUS_DNF
    assert not any(t for _, _, t, _, _ in steps[:-1])


def test_wall_contact_does_not_end_the_episode(make_env):
    env = make_env()
    steps = drive(env, lambda info: FULL_THROTTLE, 30)
    hits = [i for i, s in enumerate(steps) if s[4]["wall_contact"]]
    assert hits, "the car never touched the wall"
    assert not any(terminated or truncated for _, _, terminated, truncated, _ in steps)
    assert steps[-1][4]["status"] == STATUS_ACTIVE


def test_episode_is_truncated_at_the_step_limit(make_env):
    env = make_env(max_episode_steps=50)
    steps = drive(env, pursuit(3.0), 100)
    assert len(steps) == 50
    assert steps[-1][3] and not steps[-1][2]


def test_one_lap_is_counted_once_and_rewarded_about_1000(make_env):
    env = make_env()
    steps = drive(env, pursuit(3.0), 1200)
    laps = [i for i, s in enumerate(steps) if s[4]["lap_complete"]]
    assert len(laps) == 1 and 900 < laps[0] < 1200
    assert steps[laps[0]][4]["lap_count"] == 1
    assert sum(r for _, r, _, _, _ in steps[: laps[0] + 1]) == pytest.approx(1000, abs=5)


def test_reset_starts_a_fresh_episode(make_env):
    env = make_env()
    drive(env, pursuit(3.0), 100)
    obs, info = env.reset(seed=1)
    assert info["step_count"] == 0 and info["lap_count"] == 0
    assert info["status"] == STATUS_ACTIVE
    assert obs[102:104] == pytest.approx([0.0, 1.0])  # lap phase back to 0


def test_opponents_follow_the_opponent_policy(make_env):
    env = make_env(num_cars=2, opponent_policy=lambda obs: (0.0, 0.0))
    steps = drive(env, pursuit(3.0), 250)
    assert steps[-1][4]["status"] == STATUS_ACTIVE
    assert steps[-1][4]["step_info"]["agent_status"][1] == STATUS_DNF  # idle opponent


# --- Overrides ------------------------------------------------------------


def test_reset_options_pick_the_map(make_env):
    env = make_env(maps=["example", "Spa"])
    _, info = env.reset(seed=0, options={"map": "Spa"})
    assert info["map"] == "Spa" == es.get_current_map()


def test_friction_override_holds_during_the_episode(make_env):
    env = make_env(friction=(0.5, 0.6))
    steps = drive(env, pursuit(3.0), 60)
    mu = steps[-1][4]["friction"]
    assert 0.5 <= mu <= 0.6
    assert all(a.params["mu"] == mu for a in es._get_sim()._sim.agents)


def test_friction_changes_the_dynamics(make_env):
    def final_pose(friction):
        env = make_env(friction=friction)
        drive(env, lambda info: controls_to_action(4.0, 0.4189), 40)
        pose = es._get_sim()._sim.agents[0].state[[0, 1, 4]].copy()
        env.close()
        return pose

    assert not np.allclose(final_pose(None), final_pose(0.3), atol=1e-3)


def test_lidar_noise_override_is_multiplicative_and_bounded(make_env):
    def scans(noise):
        env = make_env(lidar_noise=noise)
        lidar = [s[4]["raw_obs"]["lidar"] for s in drive(env, pursuit(3.0), 20)]
        env.close()
        return np.array(lidar, dtype=np.float64)

    clean, noisy = scans(0.0), scans(0.1)
    ratio = noisy / clean
    assert np.all((ratio > 0.9 - 1e-6) & (ratio < 1.1 + 1e-6))
    assert ratio.std() > 0.02


# --- Singleton ------------------------------------------------------------


def test_a_second_open_env_is_refused(make_env):
    make_env()
    with pytest.raises(RuntimeError, match="singleton"):
        CrashLearnEnv(maps="example")
