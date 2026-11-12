"""Check that every delivered driver is intact and linked to how it was made.

For each submission/driver*/, the files must match the hashes in export.json and must not
be Git LFS pointers. The source model in export.json must lead, through the run manifests
under docs/results/*/manifests/, to a commit, a training configuration and a seed, and each
model it was derived or initialized from must have its own manifest. At least one result
file under docs/results/ must name the delivered model. Hashes are read from these files,
never written here: a new export only needs its manifest and its results committed.

Standard library only, so it runs on a fresh clone before any install:

    python scripts/check_deliverables.py          # add --git to also find each commit
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DELIVERED_FILES = ("agent.py", "model.onnx", "model.onnx.data")
LFS_POINTER = b"version https://git-lfs.github.com/spec/"
COMMIT = re.compile(r"[0-9a-f]{7,40}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strings(value) -> set[str]:
    """Every string value in a JSON document, at any depth."""
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return set().union(*map(strings, value)) if value else set()
    return set()


def load_results(root: Path) -> tuple[dict[str, tuple[Path, dict]], dict[Path, set[str]]]:
    """Manifests by model hash, and the string values of every other result file."""
    manifests, results = {}, {}
    for path in sorted((root / "docs" / "results").rglob("*.json")):
        document = json.loads(path.read_text())
        if path.parent.name != "manifests":
            results[path] = strings(document)
        elif "model_sha256" in document:  # an interrupted attempt has no model
            manifests.setdefault(document["model_sha256"], (path, document))
    return manifests, results


def check_lineage(model: str, manifests: dict, root: Path, errors: list, lines: list) -> None:
    """Follow derivations and initializations back to a training run."""
    seen = set()
    while model and model not in seen:
        seen.add(model)
        if model not in manifests:
            errors.append(f"model {model[:12]} has no manifest under docs/results/*/manifests/")
            return
        path, manifest = manifests[model]
        for key in ("commit", "config", "train_seed"):
            if manifest.get(key) in (None, "", {}):
                errors.append(f"{path.relative_to(root)}: no {key}")
        if not COMMIT.fullmatch(str(manifest.get("commit"))):
            errors.append(f"{path.relative_to(root)}: commit is not a Git hash")
        derivation = manifest.get("derivation") or {}
        made_by = derivation.get("method") or f"{manifest.get('timesteps_done')} steps"
        lines.append(
            f"  {model[:12]}  {path.relative_to(root)}  commit {str(manifest.get('commit'))[:12]}"
            f"  seed {manifest.get('train_seed')}  {made_by}"
        )
        if manifest.get("derivation") is not None:
            parent, field = derivation.get("source_model_sha256"), "derivation.source_model_sha256"
        elif manifest.get("initialization") is not None:
            parent = (manifest["initialization"] or {}).get("checkpoint_sha256")
            field = "initialization.checkpoint_sha256"
        else:
            return
        if not parent:
            errors.append(f"{path.relative_to(root)}: no {field}")
        model = parent


def check_driver(driver: Path, manifests: dict, results: dict, root: Path) -> list[str]:
    """Errors for one delivered driver; prints its provenance on the way."""
    name = driver.relative_to(root)
    record_path = driver / "export.json"
    if not record_path.is_file():
        return [f"{name}: no export.json"]
    record = json.loads(record_path.read_text())
    errors, lines = [], []
    recorded = record.get("files", {})
    for file in DELIVERED_FILES:
        path = driver / file
        if not path.is_file():
            errors.append(f"{name}/{file}: missing")
        elif path.read_bytes().startswith(LFS_POINTER):
            errors.append(f"{name}/{file}: Git LFS pointer instead of the file")
        elif file not in recorded:
            errors.append(f"{name}/{file}: no hash in export.json")
        elif sha256(path) != recorded[file]:
            errors.append(f"{name}/{file}: SHA-256 differs from export.json")
    if record.get("dirty") is not False:
        errors.append(f"{name}: export.json does not record a clean export (dirty is not false)")
    if not COMMIT.fullmatch(str(record.get("commit"))):
        errors.append(f"{name}: export.json has no export commit")

    model = record.get("source_model_sha256")
    if not model:
        return [*errors, f"{name}: export.json has no source_model_sha256"]
    _, manifest = manifests.get(model, (None, {}))
    if manifest and manifest.get("run_id") != record.get("source_run_id"):
        errors.append(f"{name}: source_run_id differs from the run_id of its manifest")
    named_by = sorted(
        str(path.relative_to(root)) for path, found in results.items() if model in found
    )
    if not named_by:
        errors.append(f"{name}: no result file under docs/results/ names model {model[:12]}")

    sizes = ", ".join(
        f"{file} {(driver / file).stat().st_size} B"
        for file in DELIVERED_FILES
        if (driver / file).is_file()
    )
    lines.append(f"{name}: export commit {str(record.get('commit'))[:12]}; {sizes}")
    check_lineage(model, manifests, root, errors, lines)
    lines.extend(f"  results: {path}" for path in named_by)
    print("\n".join(lines))
    return errors


def missing_commits(root: Path, manifests: dict, drivers: list[Path]) -> list[str]:
    """Commits named by the deliverables that this clone cannot find."""
    commits = {str(m.get("commit")) for _, m in manifests.values()}
    commits |= {str(json.loads((d / "export.json").read_text()).get("commit")) for d in drivers}
    return [
        f"commit {commit} is not in this clone"
        for commit in sorted(commits)
        if subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=root, capture_output=True
        ).returncode
    ]


def check(root: Path, git: bool = False) -> list[str]:
    drivers = sorted(p for p in (root / "submission").glob("driver*") if p.is_dir())
    if not drivers:
        return ["no submission/driver*/ folder"]
    manifests, results = load_results(root)
    errors = []
    for driver in drivers:
        errors += check_driver(driver, manifests, results, root)
    if git:
        exported = [d for d in drivers if (d / "export.json").is_file()]
        errors += missing_commits(root, manifests, exported)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--git", action="store_true", help="also find each commit in Git")
    args = parser.parse_args()
    errors = check(args.root.resolve(), git=args.git)
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    print("FAILED" if errors else "OK: every delivered model is intact and linked")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
