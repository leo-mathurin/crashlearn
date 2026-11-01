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
