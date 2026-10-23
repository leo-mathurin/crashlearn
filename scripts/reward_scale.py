"""Measure the reward terms of a few reference policies (E-15).

Runs whole episodes through CrashLearnEnv and sums every reward term separately, to check
their orders of magnitude before fixing the coefficients. Training tracks only.

    uv run python scripts/reward_scale.py                  # 15 training tracks, all policies
    uv run python scripts/reward_scale.py --map IMS --policy pursuit
"""

import argparse
import dataclasses
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from crashlearn.env import CrashLearnEnv, controls_to_action, sim
from crashlearn.reward import TERMS, RewardConfig
from crashlearn.tracks import train_tracks


def pursuit(env, speed=6.0, lookahead=1.5):
    """Pure pursuit on the centerline: a clean driver that never exploits anything."""
    s = sim._get_sim()
    st = s._sim.agents[env.ego_id].state
    x, y, yaw = float(st[0]), float(st[1]), float(st[4])
    w = s._waypoints
    idx = int(np.argmin((w[:, 0] - x) ** 2 + (w[:, 1] - y) ** 2))
    seg = np.hypot(*np.diff(np.roll(w, -idx, axis=0), axis=0).T)
    target = (idx + 1 + int(np.searchsorted(np.cumsum(seg), lookahead))) % len(w)
    angle = math.atan2(w[target, 1] - y, w[target, 0] - x) - yaw
    return controls_to_action(speed, math.atan2(math.sin(angle), math.cos(angle)))


POLICIES = {
    "pursuit": pursuit,
    "pursuit_slow": lambda env: pursuit(env, speed=3.0),
    "idle": lambda env: controls_to_action(0.0, 0.0),
    "reverse": lambda env: controls_to_action(-2.0, 0.0),
    "random": lambda env: env.action_space.sample(),
}


def episode(env, policy, name: str, seed: int) -> dict:
    _, info = env.reset(seed=seed, options={"map": name})
    env.action_space.seed(seed)
    sums = defaultdict(float)
    steps, terminated, truncated = 0, False, False
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(policy(env))
        steps += 1
        for k, v in info["reward_terms"].items():
            sums[k] += v
    outcome = {0: "DNF", 1: "TRUNCATED", 2: "FINISHED"}[info["status"]]
    return {
        "map": name,
        "track_length_m": round(info["track_length_m"], 1),
        "outcome": outcome,
        "steps": steps,
        "laps": info["lap_count"],
        "return": round(math.fsum(sums.values()), 2),
        "terms": {k: round(sums[k], 2) for k in TERMS},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", action="append", dest="maps", help="training track, repeatable")
    parser.add_argument("--policy", action="append", choices=POLICIES, dest="policies")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=8000, help="truncation (default 8000)")
    parser.add_argument("--output", type=Path, default=Path("runs/reward_scale.json"))
    args = parser.parse_args()
    maps = args.maps or train_tracks()
    if unknown := set(maps) - set(train_tracks()):
        parser.error(f"not training tracks: {sorted(unknown)}")

    config = RewardConfig()
    env = CrashLearnEnv(maps=maps, max_episode_steps=args.max_steps, reward_config=config)
    records = []
    try:
        for policy in args.policies or POLICIES:
            for name in maps:
                records.append(
                    {"policy": policy, **episode(env, POLICIES[policy], name, args.seed)}
                )
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"reward_config": dataclasses.asdict(config), "episodes": records}
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    short = {"vehicle_contact": "veh", "action_change": "act_chg", "wall_contact": "wall"}
    cols = [short.get(t, t) for t in TERMS]
    print(f"reward {config.version}, seed {args.seed}, means per episode over {len(maps)} maps\n")
    print(
        f"{'policy':<13}{'outcomes':<22}{'steps':>6}{'return':>9}"
        + "".join(f"{c:>9}" for c in cols)
    )
    for policy in dict.fromkeys(r["policy"] for r in records):
        rows = [r for r in records if r["policy"] == policy]
        outcomes = defaultdict(int)
        for r in rows:
            outcomes[r["outcome"]] += 1
        out = " ".join(f"{k[:3]}={v}" for k, v in sorted(outcomes.items()))
        steps = sum(r["steps"] for r in rows) / len(rows)
        ret = sum(r["return"] for r in rows) / len(rows)
        terms = [sum(r["terms"][t] for r in rows) / len(rows) for t in TERMS]
        print(
            f"{policy:<13}{out:<22}{steps:>6.0f}{ret:>9.1f}" + "".join(f"{v:>9.1f}" for v in terms)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
