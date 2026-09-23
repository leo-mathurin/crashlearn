"""Check (default) or refresh the generated fields of src/crashlearn/tracks.yaml.

Prints the per-track inventory. Aliases are input names only: they have no map directory.
Exit code 1 if tracks.yaml drifts from the vendored maps.
"""

import argparse
import sys

import yaml

from crashlearn import tracks

GEOMETRY_COLS = [
    ("len_m", "geometry_length_m"),
    ("turn_rad", "geometry_total_turning_rad"),
    ("k_p50", "geometry_curvature_abs_p50_radpm"),
    ("k_p90", "geometry_curvature_abs_p90_radpm"),
    ("k_max", "geometry_curvature_abs_max_radpm"),
    ("tight", "geometry_tight_turns"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rewrite generated fields")
    args = parser.parse_args()

    data = tracks._load()
    errors = []

    names = {t["name"] for t in data["tracks"]}
    aliases = {a: t["name"] for t in data["tracks"] for a in t.get("aliases", [])}
    dirs = set(tracks.track_dirs())
    if dirs != names:
        errors.append(f"directories {sorted(dirs)} != track names")
    if dirs & aliases.keys():
        errors.append(f"alias directories {sorted(dirs & aliases.keys())} (removed in E-12)")

    print(
        f"{'track':<14}{'split':<11}{'raceline':<9}" + "".join(f"{c:>10}" for c, _ in GEOMETRY_COLS)
    )
    for t in data["tracks"]:
        fresh = tracks.compute_generated_fields(tracks.MAPS_DIR / t["name"])
        if {k: t.get(k) for k in fresh} != fresh:
            errors.append(f"{t['name']}: generated fields out of date")
        t.update(fresh)
        raceline = "yes" if "raceline" in fresh["assets"] else "MISSING"
        cols = "".join(f"{fresh[k]:>10}" for _, k in GEOMETRY_COLS)
        print(f"{t['name']:<14}{t['split']:<11}{raceline:<9}{cols}")

    if args.write:
        # --write only refreshes generated fields: directory errors still fail
        errors = [e for e in errors if not e.endswith("generated fields out of date")]
        lines = tracks.TRACKS_YAML.read_text().splitlines(keepends=True)
        header = "".join(line for line in lines if line.startswith("#"))  # comments on top only
        body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100)
        tracks.TRACKS_YAML.write_text(header + body)
        print(f"\nwrote {tracks.TRACKS_YAML}")

    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
