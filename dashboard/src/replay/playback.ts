/** Playback clock and pose sampling of a replay. */
import { type Replay, STATUS } from "./replay";

export const SPEEDS = [0.25, 0.5, 1, 2, 4, 8] as const;

export interface Pose {
  x: number;
  y: number;
  heading: number;
}

/** Index of the last recorded frame at or before `time`. */
export function frameAt(replay: Replay, time: number): number {
  const frame = Math.floor(time / replay.simDt + 1e-6);
  return Math.min(Math.max(frame, 0), replay.frameCount - 1);
}

export function isActive(replay: Replay, frame: number, car: number): boolean {
  return replay.status[frame * replay.carCount + car] === STATUS.ACTIVE;
}

/** Catmull-Rom spline through `p1` and `p2`, with `p0` and `p3` as the points around them. */
function spline(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const a = -p0 + 3 * p1 - 3 * p2 + p3;
  const b = 2 * p0 - 5 * p1 + 4 * p2 - p3;
  return p1 + 0.5 * t * (p2 - p0 + t * (b + t * a));
}

function turnTo(from: number, to: number): number {
  const turn = to - from;
  return Math.atan2(Math.sin(turn), Math.cos(turn));
}

/**
 * Writes the pose of a car at `time` into `out`. On a recorded frame, the pose is the recorded
 * one. Between two frames where the car is ACTIVE, it is on a smooth curve through the recorded
 * poses: the recording has 20 poses per second, and straight segments between them show as a
 * small jolt at each recorded frame. Returns false when the car is inactive: the simulator has
 * then moved it away from the track, and `out` is not changed.
 */
export function samplePose(replay: Replay, time: number, car: number, out: Pose): boolean {
  const frame = frameAt(replay, time);
  if (!isActive(replay, frame, car)) return false;
  const { poses, carCount } = replay;
  const base = (frame * carCount + car) * 3;
  const x = poses[base] ?? 0;
  const y = poses[base + 1] ?? 0;
  const heading = poses[base + 2] ?? 0;
  const next = frame + 1;
  const blend = time / replay.simDt - frame;
  if (blend <= 0 || next >= replay.frameCount || !isActive(replay, next, car)) {
    out.x = x;
    out.y = y;
    out.heading = heading;
    return true;
  }
  const nextBase = (next * carCount + car) * 3;
  const nextX = poses[nextBase] ?? 0;
  const nextY = poses[nextBase + 1] ?? 0;
  const turn = turnTo(heading, poses[nextBase + 2] ?? 0);
  // The poses around the segment give its end directions. Where one is missing (start of the
  // race, car that goes inactive), the segment continues straight.
  let beforeX = 2 * x - nextX;
  let beforeY = 2 * y - nextY;
  let beforeTurn = -turn;
  if (frame > 0 && isActive(replay, frame - 1, car)) {
    const beforeBase = ((frame - 1) * carCount + car) * 3;
    beforeX = poses[beforeBase] ?? 0;
    beforeY = poses[beforeBase + 1] ?? 0;
    beforeTurn = turnTo(heading, poses[beforeBase + 2] ?? 0);
  }
  let afterX = 2 * nextX - x;
  let afterY = 2 * nextY - y;
  let afterTurn = 2 * turn;
  if (next + 1 < replay.frameCount && isActive(replay, next + 1, car)) {
    const afterBase = ((next + 1) * carCount + car) * 3;
    afterX = poses[afterBase] ?? 0;
    afterY = poses[afterBase + 1] ?? 0;
    afterTurn = turn + turnTo(poses[nextBase + 2] ?? 0, poses[afterBase + 2] ?? 0);
  }
  out.x = spline(beforeX, x, nextX, afterX, blend);
  out.y = spline(beforeY, y, nextY, afterY, blend);
  // Headings are taken as turns from the heading of `frame`: no jump at the ±π limit.
  out.heading = heading + spline(beforeTurn, 0, turn, afterTurn, blend);
  return true;
}

/**
 * Value of a per-frame channel (steering, speed) of a car at `time`, on a straight line between
 * the two recorded frames around it.
 */
export function sampleChannel(
  replay: Replay,
  channel: Float32Array,
  time: number,
  car: number,
): number {
  const frame = frameAt(replay, time);
  const value = channel[frame * replay.carCount + car] ?? 0;
  const next = frame + 1;
  const blend = time / replay.simDt - frame;
  if (blend <= 0 || next >= replay.frameCount || !isActive(replay, next, car)) return value;
  return value + ((channel[next * replay.carCount + car] ?? 0) - value) * blend;
}

/** Last pose of a car while it was ACTIVE, or false if it was never active. */
export function lastActivePose(replay: Replay, car: number, out: Pose): boolean {
  const frame = (replay.inactiveFrom[car] ?? 0) - 1;
  if (frame < 0) return false;
  const base = (frame * replay.carCount + car) * 3;
  out.x = replay.poses[base] ?? 0;
  out.y = replay.poses[base + 1] ?? 0;
  out.heading = replay.poses[base + 2] ?? 0;
  return true;
}

export class Playback {
  /** Simulated time of the playhead, in seconds. */
  time = 0;
  playing = false;
  speed = 1;

  constructor(readonly replay: Replay) {}

  get ended(): boolean {
    return this.time >= this.replay.duration;
  }

  get frame(): number {
    return frameAt(this.replay, this.time);
  }

  /** Moves the playhead by `seconds` of wall-clock time. Stops at the end of the race. */
  advance(seconds: number): void {
    if (!this.playing) return;
    this.time = Math.min(this.time + seconds * this.speed, this.replay.duration);
    if (this.ended) this.playing = false;
  }

  seek(time: number): void {
    this.time = Math.min(Math.max(time, 0), this.replay.duration);
  }

  /** Moves the playhead by a number of frames and aligns it on a frame. */
  step(frames: number): void {
    this.playing = false;
    this.seek((this.frame + frames) * this.replay.simDt);
  }

  toggle(): void {
    if (!this.playing && this.ended) this.time = 0;
    this.playing = !this.playing;
  }
}
