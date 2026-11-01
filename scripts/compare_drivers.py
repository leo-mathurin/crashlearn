"""Compare the E-18 drivers from their race records (docs/experience_e18.md).

Reads every `*.json` of `--results-dir`, plus the training manifests of `manifests/`, and prints
a Markdown report. Every failure stays in it: failed races and unfinished trainings.
"""

import argparse
import json
import sys
from pathlib import Path

from crashlearn.evaluation import (
    BASELINE,
    CANDIDATE,
    bootstrap_interval,
    driver_summary,
    paired_differences,
    scopes,
    verdict,
)


def load_records(results_dir: Path) -> list[dict]:
    records = []
    for path in sorted(results_dir.glob("*.json")):
        records += json.loads(path.read_text())
    return records


def unfinished_trainings(results_dir: Path) -> list[dict]:
    manifests = [json.loads(p.read_text()) for p in sorted(results_dir.glob("manifests/*.json"))]
    return [m for m in manifests if m["status"] != "completed"]


def interval_text(interval: tuple | None, scale: float = 1.0) -> str:
    if interval is None:
        return "seeds insuffisantes"
    return f"[{interval[0] * scale:+.1f} ; {interval[1] * scale:+.1f}]"


def rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def report(records: list[dict], unfinished: list[dict]) -> str:
    drivers = sorted({r["driver"] for r in records})
    lines = [
        "## Résultats par pilote\n",
        "| Pilote | Partition | Condition | Courses | Arrivées | Arrivées, moyenne par carte | "
        "Écart au nominal | DNF | TIMEOUT | Chrono médian (s) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    behaviour = [
        "\n## Comportement\n",
        "Moyennes par course. Contacts par 100 m parcourus. Un saut de progression est un pas où "
        "le simulateur crédite plus d'1 m au-delà de la distance roulée, comme une coupe à travers "
        "un mur. L'écart crédité moins roulé est un peu positif pour une trajectoire qui coupe les "
        "virages, et très positif pour un raccourci.\n",
        "| Pilote | Partition | Condition | Mur / 100 m | Voiture / 100 m | Immobile (pas) | "
        "Marche arrière (pas) | Variation des commandes | Sauts de progression | "
        "Crédité moins roulé (m) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for driver in drivers:
        mine = [r for r in records if r["driver"] == driver]
        for split, condition in scopes(mine):
            group = [r for r in mine if (r["split"], r["condition"]) == (split, condition)]
            s = driver_summary(group)
            nominal = [r for r in mine if (r["split"], r["condition"]) == (split, "nominal")]
            gap = "-"
            if nominal and condition != "nominal":
                loss = s["finish_rate"] - driver_summary(nominal)["finish_rate"]
                gap = f"{loss * 100:+.1f} pts"
            time = f"{s['median_time_s']:.2f}" if s["median_time_s"] is not None else "-"
            lines.append(
                f"| {driver} | {split} | {condition} | {s['races']} | "
                f"{s['finish_rate'] * 100:.1f} % | {s['finish_rate_per_map'] * 100:.1f} % | "
                f"{gap} | {s['dnf']} | {s['timeout']} | {time} |"
            )
            behaviour.append(
                f"| {driver} | {split} | {condition} | {rate(s['wall_steps_per_100m'])} | "
                f"{rate(s['vehicle_steps_per_100m'])} | {s['idle_steps_per_race']:.1f} | "
                f"{s['reverse_steps_per_race']:.1f} | {s['action_change_per_step']:.4f} | "
                f"{s['progress_jumps_per_race']:.2f} | {s['credited_minus_driven_m']:+.1f} |"
            )
    lines += behaviour

    lines += [
        f"\n## Apport de la mémoire : {CANDIDATE} moins {BASELINE}\n",
        "Intervalles bootstrap à 95 % par seed d'entraînement. Gain de chrono positif : "
        f"{CANDIDATE} plus rapide, sur les courses finies par les deux pilotes.\n",
        "| Partition | Condition | Seeds | Arrivées (pts) | Gain de chrono (s) | Conclusion |",
        "|---|---|---:|---|---|---|",
    ]
    for split, condition in scopes(records):
        group = [r for r in records if (r["split"], r["condition"]) == (split, condition)]
        paired = paired_differences(group)
        finish = bootstrap_interval(list(paired["finish_rate"].values()))
        gain = bootstrap_interval(list(paired["time_gain_s"].values()))
        lines.append(
            f"| {split} | {condition} | {len(paired['finish_rate'])} | "
            f"{interval_text(finish, 100)} | {interval_text(gain)} | {verdict(finish, gain)} |"
        )

    failures = [r for r in records if r["status"] != "FINISHED"]
    lines += [f"\n## Échecs conservés\n\n{len(failures)} courses sur {len(records)} non finies.\n"]
    if failures:
        lines += ["| Pilote | Seed d'entraînement | Carte | Condition | Seed | Statut | Tours |"]
        lines += ["|---|---:|---|---|---:|---|---:|"]
        lines += [
            f"| {r['driver']} | {r['train_seed']} | {r['map']} | {r['condition']} | {r['seed']} | "
            f"{r['status']} | {r['progress_laps']:.2f} |"
            for r in failures
        ]
    lines += [f"\n{len(unfinished)} entraînements non terminés.\n"]
    lines += [f"- {m['driver']} seed {m['train_seed']} : {m['status']}" for m in unfinished]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Markdown report, stdout by default")
    args = parser.parse_args()

    records = load_records(args.results_dir)
    if not records:
        parser.error(f"no race record in {args.results_dir}")
    text = report(records, unfinished_trainings(args.results_dir))
    if args.output:
        args.output.write_text(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
