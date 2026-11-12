"""Record race replays for the dashboard with the unchanged `vendor/simulation/sim_recorder.py`.

The recorder wants one team folder per car. This script builds that layout in a temporary
directory from `submission/<driver>` and runs the recorder, so `vendor/` and `submission/`
stay untouched. Sealed test tracks are refused (docs/tracks.md).

    uv run python scripts/record_replay.py --demo
    uv run python scripts/record_replay.py --map Sochi --drivers driver1 driver2 --seed 4
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from crashlearn import MAX_DECISIONS, SIM_DIR, tracks

ROOT = Path(__file__).resolve().parents[1]
MAX_CARS = 4
# Not a submission: a car that drives straight at constant speed, to record a real DNF.
CRASH_TEST = "crash-test"
CRASH_TEST_AGENT = '''"""Test driver of record_replay.py: straight line, constant speed."""


class Agent:
    def predict(self, obs, info):
        return 3.0, 0.0
'''

# name -> (map, drivers, seed). Validation tracks: train tracks would do as well.
DEMO = {
    "demo-solo-spielberg": ("Spielberg", ["driver1"], 7),
    "demo-4-voitures-spielberg": ("Spielberg", ["driver1", "driver2", "driver1", "driver2"], 3),
    "demo-dnf-sochi": ("Sochi", ["driver1", "driver2", "driver1", CRASH_TEST], 5),
}


def team_names(drivers: list[str]) -> list[str]:
    """One distinct folder name per car, in grid order (the recorder sorts the folders).

    The recorder keeps ten characters of each name: `1-driver1b` is the second car that
    `driver1` drives.
    """
    if len(drivers) == 1:
        return [drivers[0]]
    names = []
    for slot, driver in enumerate(drivers):
        copies = drivers[:slot].count(driver)
        label = "crash" if driver == CRASH_TEST else driver
        names.append(f"{slot + 1}-{label}{chr(ord('a') + copies) if copies else ''}")
    if any(len(name) > 10 for name in names):
        sys.exit(f"team names longer than the 10 characters of the recorder: {names}")
    return names


def build_teams(parent: Path, drivers: list[str]) -> None:
    for name, driver in zip(team_names(drivers), drivers, strict=True):
        team = parent / name
        if driver == CRASH_TEST:
            team.mkdir()
            (team / "agent.py").write_text(CRASH_TEST_AGENT)
            # The loader only checks that the file exists; this driver has no model.
            (team / "model.onnx").write_bytes(b"")
        else:
            source = ROOT / "submission" / driver
            if not (source / "agent.py").is_file():
                sys.exit(f"unknown driver: {driver} (no {source / 'agent.py'})")
            shutil.copytree(source, team)


def record(map_name: str, drivers: list[str], seed: int, laps: int, out: Path) -> None:
    canonical = tracks.canonical_name(map_name)
    if canonical in tracks.sealed_test_tracks(unseal=True):
        sys.exit(f"{canonical} is a sealed test track: record on a train or validation track")
    if not 1 <= len(drivers) <= MAX_CARS:
        sys.exit(f"1 to {MAX_CARS} drivers, got {len(drivers)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="crashlearn-teams-") as tmp:
        build_teams(Path(tmp), drivers)
        # One car: the recorder wants the flat folder of the submission itself.
        submission = Path(tmp) / team_names(drivers)[0] if len(drivers) == 1 else Path(tmp)
        command = [
            sys.executable,
            str(SIM_DIR / "sim_recorder.py"),
            *("--submission", str(submission)),
            *("--num-cars", str(len(drivers))),
            *("--map", canonical),
            *("--seed", str(seed)),
            *("--laps", str(laps)),
            *("--steps", str(MAX_DECISIONS)),
            *("--out", str(out.resolve())),
        ]
        subprocess.run(command, cwd=SIM_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--demo", action="store_true", help="record the demo replays")
    parser.add_argument("--map", default="Spielberg", help="train or validation track")
    parser.add_argument(
        "--drivers",
        nargs="+",
        default=["driver1"],
        help=f"1 to {MAX_CARS} folders of submission/, in grid order; "
        f"'{CRASH_TEST}' is a test car that drives straight and ends in DNF",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed of the LiDAR noise")
    parser.add_argument("--laps", type=int, default=3, help="laps to finish (tournament: 3)")
    parser.add_argument("--out", type=Path, help="output .npz (default: replays/<map>-<n>.npz)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "replays")
    args = parser.parse_args()

    if args.demo:
        for name, (map_name, drivers, seed) in DEMO.items():
            record(map_name, drivers, seed, args.laps, args.out_dir / f"{name}.npz")
        return
    default_name = f"{args.map}-{len(args.drivers)}-voitures-seed{args.seed}.npz"
    record(args.map, args.drivers, args.seed, args.laps, args.out or args.out_dir / default_name)


if __name__ == "__main__":
    main()
