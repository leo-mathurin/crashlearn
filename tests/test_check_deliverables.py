"""scripts/check_deliverables.py: delivered files intact and linked to their provenance."""

import importlib.util
import json
import shutil

import pytest
from conftest import ROOT

spec = importlib.util.spec_from_file_location(
    "check_deliverables", ROOT / "scripts" / "check_deliverables.py"
)
check_deliverables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_deliverables)


@pytest.fixture
def repo(tmp_path):
    """A copy of the deliverables and their results, safe to break."""
    for part in ("submission", "docs/results"):
        shutil.copytree(ROOT / part, tmp_path / part)
    return tmp_path


def source_model(repo, driver="driver1"):
    return json.loads((repo / "submission" / driver / "export.json").read_text())[
        "source_model_sha256"
    ]


def test_delivered_models_are_intact_and_linked():
    assert check_deliverables.check(ROOT) == []


def test_changed_model_file_is_reported(repo):
    with (repo / "submission/driver1/model.onnx.data").open("ab") as data:
        data.write(b"\0")
    assert check_deliverables.check(repo) == [
        "submission/driver1/model.onnx.data: SHA-256 differs from export.json"
    ]


def test_lfs_pointer_is_reported(repo):
    (repo / "submission/driver2/model.onnx").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 2918\n"
    )
    assert check_deliverables.check(repo) == [
        "submission/driver2/model.onnx: Git LFS pointer instead of the file"
    ]


def test_missing_manifest_is_reported(repo):
    model = source_model(repo)
    for path in (repo / "docs/results").rglob("manifests/*.json"):
        if json.loads(path.read_text()).get("model_sha256") == model:
            path.unlink()
    assert check_deliverables.check(repo) == [
        f"model {model[:12]} has no manifest under docs/results/*/manifests/"
    ]


def test_missing_ancestor_manifest_is_reported(repo):
    """driver1 starts from the E-18 reactive run: its manifest is part of the lineage."""
    path = repo / "docs/results/e-18/manifests/ppo-h1-seed0.json"
    model = json.loads(path.read_text())["model_sha256"]
    path.unlink()
    assert check_deliverables.check(repo) == [
        f"model {model[:12]} has no manifest under docs/results/*/manifests/"
    ]


def test_manifest_without_seed_is_reported(repo):
    path = repo / "docs/results/e-18/manifests/ppo-h4-seed1.json"
    manifest = json.loads(path.read_text())
    del manifest["train_seed"]
    path.write_text(json.dumps(manifest))
    assert check_deliverables.check(repo) == [
        "docs/results/e-18/manifests/ppo-h4-seed1.json: no train_seed"
    ]


def test_model_without_results_is_reported(repo):
    model = source_model(repo, "driver2")
    for path in (repo / "docs/results").rglob("*.json"):
        if path.parent.name != "manifests" and model in path.read_text():
            path.unlink()
    assert check_deliverables.check(repo) == [
        f"submission/driver2: no result file under docs/results/ names model {model[:12]}"
    ]


def test_dirty_export_is_reported(repo):
    path = repo / "submission/driver2/export.json"
    record = json.loads(path.read_text())
    record["dirty"] = True
    path.write_text(json.dumps(record))
    assert check_deliverables.check(repo) == [
        "submission/driver2: export.json does not record a clean export (dirty is not false)"
    ]


def test_missing_file_is_reported_with_the_other_drivers_checked(repo):
    (repo / "submission/driver1/model.onnx").unlink()
    (repo / "submission/driver2/model.onnx.data").write_bytes(b"")
    assert check_deliverables.check(repo) == [
        "submission/driver1/model.onnx: missing",
        "submission/driver2/model.onnx.data: SHA-256 differs from export.json",
    ]


def test_derivation_without_source_is_reported(repo):
    path = next(
        p
        for p in (repo / "docs/results").rglob("manifests/*.json")
        if json.loads(p.read_text()).get("model_sha256") == source_model(repo)
    )
    manifest = json.loads(path.read_text())
    del manifest["derivation"]["source_model_sha256"]
    path.write_text(json.dumps(manifest))
    assert check_deliverables.check(repo) == [
        f"{path.relative_to(repo)}: no derivation.source_model_sha256"
    ]
