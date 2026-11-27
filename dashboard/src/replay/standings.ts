/**
 * Race facts read from a replay: order of the cars, outcome of each car, values of one car at
 * one frame. A value that the file does not have is `null`: it is shown as unavailable, never
 * guessed.
 */
import { type Replay, STATUS } from "./replay";

export type CarStatus = "active" | "finished" | "dnf";

export function statusAt(replay: Replay, frame: number, car: number): CarStatus {
  const value = replay.status[frame * replay.carCount + car];
  return value === STATUS.ACTIVE ? "active" : value === STATUS.FINISHED ? "finished" : "dnf";
}

export interface Outcome {
  /** How the race of the car ended in this replay. `active`: still racing at the last frame. */
  status: CarStatus;
  /** First frame where the car is no longer active, or -1. */
  frame: number;
  /** Simulated time of the end, in seconds: the step count of the recorder times the time step. */
  time: number;
}

export function outcomes(replay: Replay): Outcome[] {
  return Array.from({ length: replay.carCount }, (_, car) => {
    const frame = replay.inactiveFrom[car] ?? replay.frameCount;
    if (frame >= replay.frameCount) return { status: "active", frame: -1, time: 0 };
    // Frame `i` is the state after `i + 1` simulation steps (`sim_recorder.py`).
    return { status: statusAt(replay, frame, car), frame, time: (frame + 1) * replay.simDt };
  });
}

function at(array: ArrayLike<number> | null, index: number): number | null {
  return array ? (array[index] ?? null) : null;
}

/**
 * Cars from first to last at a frame. Uses the ranks of the simulator when the file has them.
 * For an old file without ranks: finished cars first, then active cars, then DNF, each group
 * by laps and progress, as `viz_replay.py` does.
 */
export function orderAt(replay: Replay, frame: number): number[] {
  const cars = Array.from({ length: replay.carCount }, (_, car) => car);
  const cell = (car: number): number => frame * replay.carCount + car;
  const distance = (car: number): number =>
    (at(replay.lapCount, cell(car)) ?? 0) + (at(replay.progress, cell(car)) ?? 0);
  if (replay.ranks) {
    const { ranks } = replay;
    return cars.toSorted(
      (a, b) => (ranks[cell(a)] ?? 0) - (ranks[cell(b)] ?? 0) || distance(b) - distance(a) || a - b,
    );
  }
  const group = { finished: 0, active: 1, dnf: 2 };
  return cars.toSorted(
    (a, b) =>
      group[statusAt(replay, frame, a)] - group[statusAt(replay, frame, b)] ||
      distance(b) - distance(a) ||
      a - b,
  );
}

export interface CarValues {
  status: CarStatus;
  /** Completed laps. */
  laps: number | null;
  /** Share of the current lap, from 0 to 1, measured from the start position of the car. */
  progress: number | null;
  /** Measured speed, in m/s. */
  speed: number | null;
  /** Speed that the driver asked for, in m/s. */
  targetSpeed: number | null;
  /** Measured steering angle, in radians, positive to the left. */
  steering: number | null;
  /** Steering angle that the driver asked for, in radians. */
  targetSteering: number | null;
  /** Grip of the track, 1 on a dry track. */
  friction: number | null;
}

export function valuesAt(replay: Replay, frame: number, car: number): CarValues {
  const cell = frame * replay.carCount + car;
  return {
    status: statusAt(replay, frame, car),
    laps: at(replay.lapCount, cell),
    progress: at(replay.progress, cell),
    speed: at(replay.velocity, cell),
    targetSpeed: at(replay.actions, cell * 2),
    steering: at(replay.steering, cell),
    targetSteering: at(replay.actions, cell * 2 + 1),
    friction: at(replay.friction, frame),
  };
}

/** Full field of view of the LiDAR of the simulator: `_LIDAR_FOV`. */
const LIDAR_FOV = 2 * Math.PI;
/** Ranges at this distance are the clip of the simulator: the ray found no wall. */
export const LIDAR_MAX_RANGE_M = 15;

/**
 * Angle of each LiDAR beam relative to the heading of the car, in radians. The simulator scans
 * `fov * (n - 1) / n` (`_LIDAR_SCAN_FOV`), so that the first and the last beam do not overlap.
 */
export function beamAngles(beamCount: number): Float64Array {
  const span = (LIDAR_FOV * (beamCount - 1)) / beamCount;
  const step = beamCount > 1 ? span / (beamCount - 1) : 0;
  return Float64Array.from({ length: beamCount }, (_, beam) => -span / 2 + beam * step);
}
