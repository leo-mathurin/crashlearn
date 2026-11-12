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

/**
 * Writes the pose of a car at `time` into `out`. Between two frames where the car is ACTIVE,
 * the pose is interpolated. Returns false when the car is inactive: the simulator has then
 * moved it away from the track, and `out` is not changed.
 */
export function samplePose(replay: Replay, time: number, car: number, out: Pose): boolean {
  const frame = frameAt(replay, time);
  if (!isActive(replay, frame, car)) return false;
  const base = (frame * replay.carCount + car) * 3;
  const { poses } = replay;
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
  const nextBase = (next * replay.carCount + car) * 3;
  let turn = (poses[nextBase + 2] ?? 0) - heading;
  turn = Math.atan2(Math.sin(turn), Math.cos(turn));
  out.x = x + ((poses[nextBase] ?? 0) - x) * blend;
  out.y = y + ((poses[nextBase + 1] ?? 0) - y) * blend;
  out.heading = heading + turn * blend;
  return true;
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
