/** Chase camera: a perspective camera behind the Voiture suivie, that looks ahead of it. */
import * as THREE from "three/webgpu";

import type { Pose } from "../replay/playback";

const FOV_DEG = 58;
/** Camera position behind and above the car, and the point ahead that it looks at, in metres. */
const BACK_M = 1.9;
const UP_M = 0.85;
const LOOK_AHEAD_M = 1.1;
const LOOK_UP_M = 0.18;
/** A jump of the car longer than this is a seek, not a motion: the camera follows at once. */
const SNAP_DISTANCE_M = 8;
/** How fast the camera swings behind the car in a turn, and how fast it looks into it, per second. */
const SWING_RATE = 5;
const LOOK_RATE = 12;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 5;

export class ChaseCamera {
  readonly camera = new THREE.PerspectiveCamera(FOV_DEG, 1, 0.05, 3000);
  /** Point that the camera looks at, in Three.js world axes: just ahead of the car. */
  readonly target = new THREE.Vector3();
  /** Distance factor, changed with the wheel. */
  private zoom = 1;
  private placed = false;
  /** Directions of the camera arm and of the look point: they turn after the car, with a delay. */
  private armHeading = 0;
  private lookHeading = 0;
  private lastX = 0;
  private lastY = 0;

  get distance(): number {
    return this.camera.position.distanceTo(this.target);
  }

  /** Moves the camera closer or farther. `factor` above 1 moves it back. */
  dolly(factor: number): void {
    this.zoom = Math.min(Math.max(this.zoom * factor, MIN_ZOOM), MAX_ZOOM);
  }

  resize(width: number, height: number): void {
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  /**
   * Follows `pose`, in simulator metres. The camera stays at a fixed distance from the car: only
   * its direction eases after the heading of the car. The car thus keeps its place on screen
   * when frames are late or uneven. With `snap`, or after a jump of the car, the camera goes
   * behind the car at once.
   */
  update(seconds: number, pose: Pose, snap: boolean): void {
    const jumped = Math.hypot(pose.x - this.lastX, pose.y - this.lastY) > SNAP_DISTANCE_M;
    this.lastX = pose.x;
    this.lastY = pose.y;
    if (snap || jumped || !this.placed) {
      this.armHeading = pose.heading;
      this.lookHeading = pose.heading;
      this.placed = true;
    } else {
      this.armHeading = easeAngle(this.armHeading, pose.heading, seconds * SWING_RATE);
      this.lookHeading = easeAngle(this.lookHeading, pose.heading, seconds * LOOK_RATE);
    }
    const back = BACK_M * this.zoom;
    // Simulator (x, y, z) is (x, z, -y) in Three.js world axes.
    this.camera.position.set(
      pose.x - Math.cos(this.armHeading) * back,
      UP_M * this.zoom,
      -(pose.y - Math.sin(this.armHeading) * back),
    );
    this.target.set(
      pose.x + Math.cos(this.lookHeading) * LOOK_AHEAD_M,
      LOOK_UP_M,
      -(pose.y + Math.sin(this.lookHeading) * LOOK_AHEAD_M),
    );
    this.camera.lookAt(this.target);
  }
}

/** Turns `from` to `to` by the short way, with an exponential ease of `amount` time constants. */
function easeAngle(from: number, to: number, amount: number): number {
  const turn = Math.atan2(Math.sin(to - from), Math.cos(to - from));
  return from + turn * (1 - Math.exp(-amount));
}
