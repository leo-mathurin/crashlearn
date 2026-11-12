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
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 5;

export class ChaseCamera {
  readonly camera = new THREE.PerspectiveCamera(FOV_DEG, 1, 0.05, 3000);
  /** Point that the camera looks at, in Three.js world axes: just ahead of the car. */
  readonly target = new THREE.Vector3();
  /** Distance factor, changed with the wheel. */
  private zoom = 1;
  private placed = false;
  private readonly wantedPosition = new THREE.Vector3();
  private readonly wantedTarget = new THREE.Vector3();

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
   * Follows `pose`, in simulator metres. The camera eases to its place behind the car; with
   * `snap`, or after a jump of the car, it goes there at once.
   */
  update(seconds: number, pose: Pose, snap: boolean): void {
    const cos = Math.cos(pose.heading);
    const sin = Math.sin(pose.heading);
    // Simulator (x, y, z) is (x, z, -y) in Three.js world axes.
    this.wantedPosition.set(
      pose.x - cos * BACK_M * this.zoom,
      UP_M * this.zoom,
      -(pose.y - sin * BACK_M * this.zoom),
    );
    this.wantedTarget.set(pose.x + cos * LOOK_AHEAD_M, LOOK_UP_M, -(pose.y + sin * LOOK_AHEAD_M));
    const jumped = this.wantedTarget.distanceTo(this.target) > SNAP_DISTANCE_M;
    if (snap || jumped || !this.placed) {
      this.camera.position.copy(this.wantedPosition);
      this.target.copy(this.wantedTarget);
      this.placed = true;
    } else {
      // The look point follows faster than the body of the camera: the car stays framed.
      this.camera.position.lerp(this.wantedPosition, 1 - Math.exp(-seconds * 7));
      this.target.lerp(this.wantedTarget, 1 - Math.exp(-seconds * 16));
    }
    this.camera.lookAt(this.target);
  }
}
