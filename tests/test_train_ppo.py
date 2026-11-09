"""E-18 chain end to end, out of CI: the CI has no torch (`uv sync --group train` here)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")
ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow


def run(script: str, *args) -> subprocess.CompletedProcess:
    command = [sys.executable, str(ROOT / "scripts" / script), *map(str, args)]
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True)


def train(run_dir: Path, *extra) -> subprocess.CompletedProcess:
    return run(
        "train_ppo.py",
        *("--history-depth", 4, "--seed", 0, "--timesteps", 4096, "--n-envs", 2),
        *("--checkpoint-every", 4096, "--output-dir", run_dir),
        *extra,
    )


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> Path:
    run_dir = tmp_path_factory.mktemp("run")
    result = train(run_dir)
    assert result.returncode == 0, result.stderr
    return run_dir


def race(run_dir: Path, output: Path) -> list[dict]:
    result = run(
        "evaluate_drivers.py",
        *("--run-dir", run_dir, "--output", output, "--map", "IMS"),
        *("--condition", "nominal", "--seed", 0, "--max-steps", 60),
    )
    assert result.returncode == 0, result.stderr
    return json.loads(output.read_text())


def test_a_run_records_what_produced_it(trained):
    manifest = json.loads((trained / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["driver"] == "ppo-h4"
    assert manifest["timesteps_done"] >= 4096
    assert len(manifest["model_sha256"]) == 64


def test_an_existing_run_is_not_overwritten(trained):
    before = (trained / "manifest.json").read_text()
    result = train(trained)
    assert result.returncode == 2
    assert (trained / "manifest.json").read_text() == before


def test_evaluating_a_model_twice_gives_the_same_races(trained, tmp_path):
    first = race(trained, tmp_path / "a.json")
    second = race(trained, tmp_path / "b.json")
    assert first[0]["history_depth"] == 4
    assert first[0]["trajectory_sha256"] == second[0]["trajectory_sha256"]


def test_a_modified_model_is_refused(trained, tmp_path):
    copy = tmp_path / "run"
    copy.mkdir()
    (copy / "manifest.json").write_text((trained / "manifest.json").read_text())
    (copy / "model.zip").write_bytes((trained / "model.zip").read_bytes() + b"x")
    result = run("evaluate_drivers.py", "--run-dir", copy, "--output", tmp_path / "r.json")
    assert result.returncode != 0
    assert "does not match its manifest" in result.stderr


def test_a_failed_training_keeps_its_manifest(tmp_path):
    result = train(tmp_path, "--device", "not-a-device")
    assert result.returncode != 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "not-a-device" in manifest["error"]


def test_resume_keeps_prior_logs_and_reloads_a_periodic_checkpoint(tmp_path):
    run_dir = tmp_path / "resume"
    first = train(run_dir, "--wandb")
    assert first.returncode == 0, first.stderr
    before = json.loads((run_dir / "manifest.json").read_text())
    log = run_dir / before["attempts"][0]["log_dir"] / "progress.csv"
    original_log = log.read_bytes()
    checkpoint = next((run_dir / "checkpoints").glob("*.zip"))
    resumed = train(run_dir, "--resume", checkpoint, "--timesteps", 8192, "--wandb")
    assert resumed.returncode == 0, resumed.stderr
    after = json.loads((run_dir / "manifest.json").read_text())
    assert after["run_id"] == before["run_id"]
    assert len(after["attempts"]) == 2
    assert after["attempts"][0] == before["attempts"][0]
    assert after["attempts"][1]["first_step"] == 4096
    assert after["timesteps_done"] >= 8192
    assert log.read_bytes() == original_log
    assert after["attempts"][0]["wandb"]["mode"] == "offline"
    assert list(run_dir.glob("logs/attempt-001/wandb/offline-run-*/*.wandb"))


def test_resume_refuses_changed_preprocessing_without_rewriting_the_run(trained):
    path = trained / "manifest.json"
    original = path.read_text()
    modified = json.loads(original)
    modified["features_sha256"] = "different"
    path.write_text(json.dumps(modified))
    before = path.read_bytes()
    try:
        checkpoint = next((trained / "checkpoints").glob("*.zip"))
        result = train(trained, "--resume", checkpoint, "--timesteps", 8192)
        assert result.returncode == 2
        assert "features_sha256" in result.stderr
        assert path.read_bytes() == before
    finally:
        path.write_text(original)


def test_exported_agent_matches_the_policy_on_real_observations(trained, tmp_path):
    import importlib.util

    import numpy as np
    from stable_baselines3 import PPO

    from crashlearn.env import CrashLearnEnv, action_to_controls
    from crashlearn.history_env import HistoryWrapper

    folder = tmp_path / "driver2"
    result = run("export_agent.py", "--run-dir", trained, "--output-dir", folder)
    assert result.returncode == 0, result.stderr
    checked = run("../vendor/simulation/verify_submission.py", "--submission", folder)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    spec = importlib.util.spec_from_file_location("exported_agent", folder / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = module.Agent()
    policy = PPO.load(trained / "model.zip", device="cpu")
    env = HistoryWrapper(CrashLearnEnv(maps="IMS"), 4)
    try:
        obs, info = env.reset(seed=1)
        for _ in range(50):
            expected = policy.predict(obs, deterministic=True)[0]
            controls = agent.predict(info["raw_obs"], info["step_info"])
            np.testing.assert_allclose(controls, action_to_controls(expected), atol=6e-5, rtol=0)
            obs, _, terminated, truncated, info = env.step(expected)
            if terminated or truncated:
                obs, info = env.reset(seed=1)
    finally:
        env.close()


def test_wandb_failure_does_not_stop_local_logging(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import wandb
    from stable_baselines3.common.logger import configure

    from crashlearn.experiments import attach_wandb

    def unavailable(*args, **kwargs):
        raise ConnectionError("offline")

    remote = SimpleNamespace(id="test", url=None, log=unavailable, finish=unavailable)
    monkeypatch.setattr(wandb, "init", lambda **kwargs: remote)
    manifest = {
        "run_id": "test",
        "config": {},
        "train_seed": 0,
        "history_depth": 1,
        "commit": "test",
        "versions": {},
        "attempts": [{}],
    }
    args = SimpleNamespace(
        wandb_project="test",
        wandb_entity=None,
        output_dir=tmp_path,
        wandb_group=None,
        wandb_mode="offline",
        log_dir=tmp_path,
    )
    logger = configure(str(tmp_path), ["csv"])
    attach_wandb(logger, args, manifest)
    logger.record("reward", 1.0)
    logger.dump(step=10)
    logger.record("reward", 2.0)
    logger.dump(step=20)
    logger.close()
    assert (tmp_path / "progress.csv").read_text().splitlines() == ["reward", "1.0", "2.0"]
    assert manifest["attempts"][0]["wandb_error"] == "offline"
