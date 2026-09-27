#!/usr/bin/env python3
"""Mesure la largeur de piste a chaque waypoint et reecrit les colonnes
w_tr_right_m / w_tr_left_m des *_centerline.csv.

Les fichiers amont portent 1.1 m de chaque cote sur tous les circuits, alors que le
couloir reellement dessine va de 1.44 m (Montreal) a 2.60 m (Brands Hatch).

Usage :
    python3 measure_widths.py [--dry-run] [circuit ...]
"""

import argparse
import glob
import os

import numpy as np
import yaml
from PIL import Image

STEP_FRAC = 0.25      # pas du rayon, en fraction de pixel
MAX_RAY_M = 6.0       # distance max sondee (m)
FREE_LEVEL = 128      # niveau de gris >= : cellule libre
MEDIAN_WIN = 9        # fenetre du filtre median (impair)
CLAMP_PCT = 95.0      # percentile de clamp haut, par cote


def _load(track_dir):
    yaml_path = glob.glob(os.path.join(track_dir, "*_map.yaml"))[0]
    csv_path = glob.glob(os.path.join(track_dir, "*_centerline.csv"))[0]
    meta = yaml.safe_load(open(yaml_path))
    img = np.array(Image.open(os.path.join(track_dir, meta["image"])).convert("L"))
    origin = meta["origin"]
    return img, float(meta["resolution"]), float(origin[0]), float(origin[1]), csv_path


def _normals(xs, ys):
    """Normale unitaire droite (convention du simulateur : lateral > 0 = droite)."""
    dx = np.roll(xs, -1) - np.roll(xs, 1)      # boucle fermee
    dy = np.roll(ys, -1) - np.roll(ys, 1)
    norm = np.hypot(dx, dy)
    norm[norm == 0.0] = 1.0
    return dy / norm, -dx / norm


def _ray(img, res, ox, oy, x, y, nx, ny):
    """Distance (m) jusqu'au premier pixel occupe dans la direction (nx, ny)."""
    h, w = img.shape
    step = res * STEP_FRAC
    t = 0.0
    while t < MAX_RAY_M:
        t += step
        col = int(round((x + nx * t - ox) / res))
        row = int(round(h - 1 - (y + ny * t - oy) / res))
        if col < 0 or row < 0 or col >= w or row >= h or img[row, col] < FREE_LEVEL:
            return t - step * 0.5
    return MAX_RAY_M


def _median_filter(v, win):
    half = win // 2
    idx = (np.arange(len(v))[:, None] + np.arange(-half, half + 1)[None, :]) % len(v)
    return np.median(v[idx], axis=1)


def _smooth(v, win=5):
    half = win // 2
    idx = (np.arange(len(v))[:, None] + np.arange(-half, half + 1)[None, :]) % len(v)
    return v[idx].mean(axis=1)


def _sides(img, res, ox, oy, xs, ys, nx, ny):
    right = np.array([_ray(img, res, ox, oy, xs[i], ys[i], nx[i], ny[i]) for i in range(len(xs))])
    left = np.array([_ray(img, res, ox, oy, xs[i], ys[i], -nx[i], -ny[i]) for i in range(len(xs))])
    return right, left


def build_example(here, iters=6):
    """Genere examples/example_centerline.csv depuis la trajectoire optimale.

    example n'a pas de centerline : son seul fichier de waypoints est une raceline
    TUM, decentree de 1.50 m en mediane. On recentre chaque point de (R-L)/2 le long
    de la normale, en iterant, puis on mesure les largeurs sur la ligne obtenue.
    """
    d = os.path.join(here, "examples")
    meta = yaml.safe_load(open(os.path.join(d, "example_map.yaml")))
    img = np.array(Image.open(os.path.join(d, meta["image"])).convert("L"))
    res, ox, oy = float(meta["resolution"]), float(meta["origin"][0]), float(meta["origin"][1])

    rl = np.loadtxt(os.path.join(d, "example_waypoints.csv"), delimiter=";", skiprows=3)
    xs, ys = rl[:, 1].copy(), rl[:, 2].copy()

    for _ in range(iters):
        nx, ny = _normals(xs, ys)
        right, left = _sides(img, res, ox, oy, xs, ys, nx, ny)
        shift = _median_filter((right - left) / 2.0, MEDIAN_WIN)
        xs, ys = _smooth(xs + nx * shift), _smooth(ys + ny * shift)

    nx, ny = _normals(xs, ys)
    right, left = _sides(img, res, ox, oy, xs, ys, nx, ny)
    right = np.minimum(_median_filter(right, MEDIAN_WIN), np.percentile(right, CLAMP_PCT))
    left = np.minimum(_median_filter(left, MEDIAN_WIN), np.percentile(left, CLAMP_PCT))
    return os.path.join(d, "example_centerline.csv"), xs, ys, right, left


def measure(track_dir):
    img, res, ox, oy, csv_path = _load(track_dir)
    cl = np.loadtxt(csv_path, delimiter=",", comments="#")
    xs, ys = cl[:, 0], cl[:, 1]
    nx, ny = _normals(xs, ys)

    right = np.empty(len(xs))
    left = np.empty(len(xs))
    for i in range(len(xs)):
        right[i] = _ray(img, res, ox, oy, xs[i], ys[i], nx[i], ny[i])
        left[i] = _ray(img, res, ox, oy, xs[i], ys[i], -nx[i], -ny[i])

    right = _median_filter(right, MEDIAN_WIN)
    left = _median_filter(left, MEDIAN_WIN)
    # Les ouvertures de mur (stands) laissent fuir le rayon.
    right = np.minimum(right, np.percentile(right, CLAMP_PCT))
    left = np.minimum(left, np.percentile(left, CLAMP_PCT))
    return csv_path, xs, ys, right, left


def write_csv(csv_path, xs, ys, right, left):
    lines = ["# x_m, y_m, w_tr_right_m, w_tr_left_m\n"]
    for x, y, r, l in zip(xs, ys, right, left):
        lines.append(f"{x!r}, {y!r}, {r:.4f}, {l:.4f}\n")
    with open(csv_path, "w") as f:
        f.writelines(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", nargs="*", help="noms de circuits (defaut : tous)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--example", action="store_true",
                    help="genere examples/example_centerline.csv et s'arrete")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))

    if args.example:
        csv_path, xs, ys, right, left = build_example(here)
        total = right + left
        print(f"example {len(xs)} pts : couloir min={total.min():.2f} med={np.median(total):.2f} "
              f"max={total.max():.2f} | asymetrie med={np.median(np.abs(right - left)):.2f}")
        if not args.dry_run:
            write_csv(csv_path, xs, ys, right, left)
            print(f"ecrit -> {csv_path}")
        return

    if args.tracks:
        dirs = [os.path.join(here, t) for t in args.tracks]
    else:
        dirs = sorted(
            os.path.dirname(p) for p in glob.glob(os.path.join(here, "*", "*_centerline.csv"))
        )

    print(f"{'circuit':16s} {'n':>5s} {'couloir min':>12s} {'med':>6s} {'max':>6s}")
    for d in dirs:
        csv_path, xs, ys, right, left = measure(d)
        total = right + left
        print(
            f"{os.path.basename(d):16s} {len(xs):5d} "
            f"{total.min():12.2f} {np.median(total):6.2f} {total.max():6.2f}"
        )
        if not args.dry_run:
            write_csv(csv_path, xs, ys, right, left)


if __name__ == "__main__":
    main()
