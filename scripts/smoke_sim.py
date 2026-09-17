"""Headless smoke run of the vendored simulator: one car, trivial policy, no RL."""

import argparse
import sys
import traceback

from crashlearn import add_simulator_to_path

STATUS_NAMES = {0: "DNF", 1: "ACTIVE", 2: "FINISHED"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="Austin", help="circuit name (see get_available_maps)")
    parser.add_argument("--steps", type=int, default=200, help="decision steps (20 Hz)")
    parser.add_argument("--speed", type=float, default=3.0, help="constant target speed (m/s)")
    parser.add_argument(
        "--steer-gain",
        type=float,
        default=0.0,
        help="steering = gain * (mean left lidar - mean right lidar); 0 keeps wheels straight",
    )
    args = parser.parse_args()

    add_simulator_to_path()
    import env_simulation as sim

    maps = sim.get_available_maps()
    if args.map not in maps:
        print(f"unknown map {args.map!r}; available: {', '.join(maps)}", file=sys.stderr)
        return 2

    try:
        sim.set_map(args.map)
        sim.reset(num_cars=1)
        for _ in range(args.steps):
            obs = sim.get_obs(0)
            lidar = obs["lidar"]
            half = len(lidar) // 2
            # ponytail: assumes rays ordered right-to-left (f110 convention), unverified
            steering = args.steer_gain * float(lidar[half:].mean() - lidar[:half].mean())
            sim.apply_action(0, args.speed, steering)
            sim.simulation_step()

        info = sim.get_step_info()
        obs = sim.get_obs(0)
        print(f"map              : {args.map}")
        print(f"step_count       : {info['step_count']}")
        print(f"time_elapsed     : {info['time_elapsed']:.2f} s")
        print(f"progress         : {obs['progress']:.4f}")
        print(f"collisions       : wall={info['collisions']['wall'][0]}")
        print(f"                   vehicle={info['collisions']['vehicle'][0]}")
        print(f"friction_current : {info['friction_current']}")
        status = info["agent_status"][0]
        print(f"agent_status     : {status} ({STATUS_NAMES.get(status, '?')})")
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        sim.close()


if __name__ == "__main__":
    sys.exit(main())
