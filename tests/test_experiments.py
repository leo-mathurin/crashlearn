"""Run records: provenance, frozen resumes and the final status of every attempt."""

import json

import pytest

from crashlearn.experiments import Run, provenance

CONFIG = {"n_envs": 2, "fixed_map_workers": False, "max_speed": 10.0, "random_start": False}


def open_run(directory, resume=None, **config):
    return Run.open(directory, history_depth=1, seed=0, config=CONFIG | config, resume=resume)


def manifest(run_dir):
    return json.loads((run_dir / "manifest.json").read_text())


def test_provenance_pins_the_inputs_of_a_run():
    assert {"commit", "dirty", "sim_sha256", "features_sha256", "reward_sha256"} <= set(
        provenance()
    )


def test_a_failed_attempt_keeps_its_error(tmp_path):
    run = open_run(tmp_path)
    with pytest.raises(RuntimeError), run.attempt(budget_timesteps=10, resumed_from=None):
        raise RuntimeError("no GPU")
    record = manifest(tmp_path)
    assert record["status"] == record["attempts"][0]["status"] == "failed"
    assert "no GPU" in record["error"]
    assert record["attempts"][0]["finished_utc"]


def test_an_interruption_is_not_a_failure(tmp_path):
    run = open_run(tmp_path)
    with pytest.raises(KeyboardInterrupt), run.attempt(budget_timesteps=10, resumed_from=None):
        raise KeyboardInterrupt
    assert manifest(tmp_path)["status"] == "interrupted"


@pytest.mark.parametrize("training_fails", [False, True])
def test_a_cleanup_failure_still_writes_the_final_manifest(tmp_path, training_fails):
    run = open_run(tmp_path)
    closed = []

    def fail():
        raise RuntimeError("logger cleanup failure")

    with pytest.raises(RuntimeError, match="cleanup failure"):
        with run.attempt(budget_timesteps=10, resumed_from=None) as cleanups:
            cleanups.callback(closed.append, "env")
            cleanups.callback(fail)
            if training_fails:
                raise RuntimeError("training failure")
    attempt = manifest(tmp_path)["attempts"][-1]
    assert attempt["status"] == "failed" and attempt["finished_utc"]
    assert "logger cleanup failure" in attempt["cleanup_error"]
    assert closed == ["env"]


def test_a_new_run_never_replaces_an_existing_folder(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="not empty"):
        open_run(tmp_path)


@pytest.mark.parametrize("change", [{"max_speed": 5.0}, {"random_start": True}, {"n_envs": 4}])
def test_a_resume_refuses_another_training_distribution(tmp_path, change):
    run = open_run(tmp_path)
    run.manifest["attempts"].append({})
    run.save()
    checkpoint = tmp_path / "checkpoints" / "ppo.zip"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"")
    with pytest.raises(ValueError, match=next(iter(change))):
        open_run(tmp_path, resume=checkpoint, **change)
