"""E-57 training inputs: progress-blind frames and feature dropout (InputMaskWrapper)."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from crashlearn.features import EGO_SIZE, FEATURE_SIZE
from crashlearn.history_env import PROGRESS_COLUMNS, InputMaskWrapper


class Constant(gym.Env):
    """Always the same observation, all ones: what the mask zeroes is plain to see."""

    observation_space = spaces.Box(-1.0, 1.0, (FEATURE_SIZE,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.ones(FEATURE_SIZE, dtype=np.float32), {}

    def step(self, action):
        return np.ones(FEATURE_SIZE, dtype=np.float32), 0.0, False, False, {}


def test_progress_blind_zeroes_only_the_progress():
    env = InputMaskWrapper(Constant(), progress_blind=True)
    obs, _ = env.reset(seed=0)
    for frame in (obs, env.step(np.zeros(2))[0]):
        assert not frame[PROGRESS_COLUMNS].any()
        kept = np.delete(frame, np.arange(FEATURE_SIZE)[PROGRESS_COLUMNS])
        assert (kept == 1).all()


def test_feature_dropout_rate_scope_and_seed():
    env = InputMaskWrapper(Constant(), feature_dropout=0.2)
    env.reset(seed=3)
    frames = np.array([env.step(np.zeros(2))[0] for _ in range(500)])
    assert (frames[:, EGO_SIZE:] == 1).all()  # opponent features untouched
    assert (frames[:, :EGO_SIZE] == 0).mean() == pytest.approx(0.2, abs=0.01)
    env.reset(seed=3)
    again = np.array([env.step(np.zeros(2))[0] for _ in range(500)])
    assert np.array_equal(frames, again)  # the same seed replays the same masks


def test_mask_is_off_by_default_and_checks_its_rate():
    env = InputMaskWrapper(Constant())
    assert (env.reset(seed=0)[0] == 1).all()
    with pytest.raises(ValueError):
        InputMaskWrapper(Constant(), feature_dropout=1.0)


def test_blind_progress_zeroes_every_stacked_frame():
    import importlib.util
    from pathlib import Path

    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    script = Path(__file__).resolve().parents[1] / "scripts" / "train_ppo.py"
    spec = importlib.util.spec_from_file_location("train_ppo", script)
    train_ppo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_ppo)

    class Stacked(Constant):
        observation_space = spaces.Box(-1.0, 1.0, (4 * FEATURE_SIZE,), dtype=np.float32)

    model = PPO("MlpPolicy", Stacked(), n_steps=8, batch_size=8, device="cpu")
    train_ppo.blind_progress(model, 4)
    columns = train_ppo.progress_columns(4)
    assert len(columns) == 4 * 3 and columns[3] == FEATURE_SIZE + PROGRESS_COLUMNS.start
    for network in (model.policy.mlp_extractor.policy_net, model.policy.mlp_extractor.value_net):
        weight = network[0].weight.detach()
        assert not weight[:, columns].any()
        others = torch.ones(weight.shape[1], dtype=torch.bool)
        others[columns] = False
        assert weight[:, others].abs().sum() > 0
