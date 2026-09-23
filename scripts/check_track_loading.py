"""Really load every map (set_map, reset, a few straight decisions) and report OK/FAIL.

Headless load check, not a policy evaluation. Exit code 1 if a canonical track fails;
alias failures are reported but tolerated (simulator issue, see docs/tracks.md).
"""

import argparse
import sys
import time

from crashlearn import add_simulator_to_path, tracks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20, help="decision steps (20 Hz)")
    args = parser.parse_args()

    add_simulator_to_path()
    import env_simulation as sim

    canonical = set(tracks.final_retraining_tracks())
    names = sorted(set(sim.get_available_maps()) | canonical)
    failed = {"track": [], "alias": []}
    try:
        for name in names:
            kind = "track" if name in canonical else "alias"
            start = time.perf_counter()
            try:
                r = tracks.load_and_step(name, args.steps)
            except Exception as e:
                failed[kind].append(name)
                print(f"FAIL {kind} {name!r}: {type(e).__name__}: {e}")
                continue
            print(
                f"OK   {kind} {name!r}: time={r['time']:.2f}s progress={r['progress']:.4f} "
                f"status={r['status']} wall={r['wall']} ({time.perf_counter() - start:.2f}s)"
            )
    finally:
        sim.close()

    print(
        f"\n{len(names)} names: {len(failed['track'])} canonical failures, "
        f"{len(failed['alias'])} alias failures"
    )
    return 1 if failed["track"] else 0


if __name__ == "__main__":
    sys.exit(main())
