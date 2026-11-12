"""Raw results leave Git; run manifests stay (scripts/results_artifact.py)."""

import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "results_artifact", ROOT / "scripts/results_artifact.py"
)
results_artifact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(results_artifact)


def test_only_raw_results_leave_git(tmp_path):
    for name in ("race.json", "selection/agents.json", "manifests/run.json", "README.md"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("{}")
    names = [p.relative_to(tmp_path).as_posix() for p in results_artifact.results(tmp_path)]
    assert names == ["race.json", "selection/agents.json"]


def test_git_ignores_raw_results_but_keeps_manifests():
    def ignored(path: str) -> bool:
        command = ["git", "check-ignore", "-q", "--no-index", path]
        return subprocess.run(command, cwd=ROOT).returncode == 0

    assert ignored("docs/results/e-18/ppo-h1-seed0.json")
    assert ignored("docs/results/e-18/selection/agents.json")
    assert not ignored("docs/results/e-18/manifests/ppo-h1-seed0.json")
    assert not ignored("docs/results/e-18/SHA256SUMS")


def test_every_campaign_with_sums_names_an_artifact_version():
    for sums in ROOT.glob("docs/results/*/SHA256SUMS"):
        header, *entries = sums.read_text().splitlines()
        assert header.startswith("# crashlearn/crashlearn/results-"), sums
        assert ":v" in header and entries, sums
        assert all(len(line.split("  ", 1)[0]) == 64 for line in entries), sums
