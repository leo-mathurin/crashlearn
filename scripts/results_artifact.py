"""Keep raw result JSON files in W&B, outside Git (docs/experiments.md, "Résultats bruts").

`publish docs/results/<campaign>` uploads every JSON file of the folder, except the run
manifests that stay in Git, as the artifact `results-<campaign>` of the shared project. It
writes `SHA256SUMS` next to the README: Git keeps the exact content that the docs cite.
`fetch` downloads the version that `SHA256SUMS` names and checks every file against it.
"""

import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT = "crashlearn/crashlearn"
SUMS = "SHA256SUMS"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def results(folder: Path) -> list[Path]:
    """The raw results: JSON files outside `manifests/`, which stay in Git."""
    return sorted(
        p for p in folder.rglob("*.json") if "manifests" not in p.relative_to(folder).parts
    )


def publish(folder: Path) -> int:
    import wandb

    files = results(folder)
    if not files:
        print(f"no result JSON in {folder}", file=sys.stderr)
        return 1
    name = f"results-{folder.name}"
    with tempfile.TemporaryDirectory() as staging:
        for path in files:
            target = Path(staging) / path.relative_to(folder)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        artifact = wandb.Artifact(name, type="results")
        artifact.add_dir(staging)
        run = wandb.init(
            project=PROJECT.split("/")[1],
            entity=PROJECT.split("/")[0],
            job_type="publish-results",
            name=f"publish-{name}",
        )
        logged = run.log_artifact(artifact)
        logged.wait()
        run.finish()
    lines = [f"# {PROJECT}/{name}:{logged.version}"]
    lines += [f"{sha256(p)}  {p.relative_to(folder).as_posix()}" for p in files]
    (folder / SUMS).write_text("\n".join(lines) + "\n")
    print(f"{PROJECT}/{name}:{logged.version}, {len(files)} files; wrote {folder / SUMS}")
    return 0


def fetch(folder: Path) -> int:
    import wandb

    header, *entries = (folder / SUMS).read_text().splitlines()
    reference = header.removeprefix("# ")
    wandb.Api().artifact(reference).download(root=str(folder))
    bad = []
    for line in entries:
        digest, name = line.split("  ", 1)
        path = folder / name
        if not path.is_file() or sha256(path) != digest:
            bad.append(name)
    if bad:
        print(f"{len(bad)} files differ from {SUMS}: {', '.join(bad)}", file=sys.stderr)
        return 1
    print(f"{reference}: {len(entries)} files match {SUMS}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("publish", "fetch"))
    parser.add_argument("folder", type=Path, help="a docs/results/<campaign> folder")
    args = parser.parse_args()
    return (publish if args.action == "publish" else fetch)(args.folder)


if __name__ == "__main__":
    raise SystemExit(main())
