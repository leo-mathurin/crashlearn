/**
 * Free camera: a perspective camera in orbit around a point on the ground. Orbit, zoom and
 * move with the mouse, the trackpad or touch; a double-click moves the orbit point. The keys
 * Z, Q, S, D (W, A, S, D on a QWERTY keyboard) move the camera over the ground.
 */
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import * as THREE from "three/webgpu";

const FOV_DEG = 42;
/** Polar angle of the first view, from the vertical: a three-quarter view from the south. */
const FIRST_POLAR_RAD = (52 * Math.PI) / 180;

/**
 * Move keys by their place on the keyboard, not by their letter: the same four keys are Z, Q,
 * S, D on an AZERTY keyboard and W, A, S, D on a QWERTY one. Each gives (forward, right).
 */
const MOVE_KEYS: Record<string, readonly [number, number]> = {
  KeyW: [1, 0],
  KeyS: [-1, 0],
  KeyA: [0, -1],
  KeyD: [0, 1],
};
/** Speed of a key move, in orbit distances per second: the same pace on screen at any zoom. */
const MOVE_SPEED = 0.9;
const FAST_FACTOR = 2.5;
/** How fast a key move starts and stops, per second. */
const MOVE_EASE_RATE = 12;

export class FreeCamera {
  readonly camera = new THREE.PerspectiveCamera(FOV_DEG, 1, 0.1, 4000);
  private readonly controls: OrbitControls;
  private readonly raycaster = new THREE.Raycaster();
  private readonly ground = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  /** Point that the orbit centre moves to after a double-click, or null. */
  private orbitGoal: THREE.Vector3 | null = null;
  private readonly scratch = new THREE.Vector3();
  private readonly heldKeys = new Set<string>();
  /** With Shift, a key move is faster. */
  fast = false;
  /** Current speed of the key move over the ground, in metres per second (Three.js x and z). */
  private readonly velocity = new THREE.Vector2();
  private readonly forward = new THREE.Vector3();
  /** False until the first view is set: the camera then frames what the Aerial camera shows. */
  placed = false;

  constructor(private readonly element: HTMLElement) {
    this.controls = new OrbitControls(this.camera, element);
    this.controls.enabled = false;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.zoomToCursor = true;
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 600;
    // Never below the ground.
    this.controls.maxPolarAngle = (86 * Math.PI) / 180;
  }

  set enabled(value: boolean) {
    this.controls.enabled = value;
    if (!value) this.releaseKeys();
  }

  /** Holds a move key down. Returns false when `code` is not a move key. */
  pressKey(code: string): boolean {
    if (!(code in MOVE_KEYS)) return false;
    this.heldKeys.add(code);
    return true;
  }

  releaseKey(code: string): void {
    this.heldKeys.delete(code);
  }

  releaseKeys(): void {
    this.heldKeys.clear();
  }

  /** Orbit centre on the ground, in Three.js world axes. */
  get target(): THREE.Vector3 {
    return this.controls.target;
  }

  get distance(): number {
    return this.camera.position.distanceTo(this.controls.target);
  }

  /** Looks at `target` from the south, far enough to show `halfHeight` metres around it. */
  frame(target: THREE.Vector3, halfHeight: number): void {
    const distance = halfHeight / Math.tan(THREE.MathUtils.degToRad(FOV_DEG / 2));
    this.orbitGoal = null;
    this.velocity.set(0, 0);
    this.controls.target.copy(target);
    this.camera.position
      .set(0, Math.cos(FIRST_POLAR_RAD), Math.sin(FIRST_POLAR_RAD))
      .multiplyScalar(distance)
      .add(target);
    this.settle();
    this.placed = true;
  }

  /** Makes the ground point under a screen position the new centre of the orbit. */
  orbitAround(clientX: number, clientY: number): void {
    const box = this.element.getBoundingClientRect();
    const pointer = new THREE.Vector2(
      ((clientX - box.left) / box.width) * 2 - 1,
      -((clientY - box.top) / box.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(pointer, this.camera);
    const hit = this.raycaster.ray.intersectPlane(this.ground, new THREE.Vector3());
    if (hit) this.orbitGoal = hit;
  }

  resize(width: number, height: number): void {
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  /** Moves the camera and its orbit centre over the ground, for the move keys held down. */
  private moveWithKeys(seconds: number): void {
    let ahead = 0;
    let right = 0;
    if (this.controls.enabled) {
      for (const code of this.heldKeys) {
        ahead += MOVE_KEYS[code]?.[0] ?? 0;
        right += MOVE_KEYS[code]?.[1] ?? 0;
      }
    }
    const { velocity, forward } = this;
    if (ahead === 0 && right === 0 && velocity.lengthSq() < 1e-6) return;
    // "Forward" is where the camera looks, flat on the ground.
    forward.subVectors(this.controls.target, this.camera.position).setY(0);
    if (forward.lengthSq() < 1e-9) this.camera.getWorldDirection(forward).setY(0);
    forward.normalize();
    const length = Math.hypot(ahead, right) || 1;
    const speed = Math.max(this.distance, 2) * MOVE_SPEED * (this.fast ? FAST_FACTOR : 1);
    // The right of the view is the forward direction turned by a quarter turn.
    const wantedX = ((forward.x * ahead - forward.z * right) / length) * speed;
    const wantedZ = ((forward.z * ahead + forward.x * right) / length) * speed;
    const ease = 1 - Math.exp(-seconds * MOVE_EASE_RATE);
    velocity.x += (wantedX - velocity.x) * ease;
    velocity.y += (wantedZ - velocity.y) * ease;
    if (ahead !== 0 || right !== 0) this.orbitGoal = null;
    this.scratch.set(velocity.x * seconds, 0, velocity.y * seconds);
    this.controls.target.add(this.scratch);
    this.camera.position.add(this.scratch);
  }

  update(seconds: number): void {
    this.moveWithKeys(seconds);
    if (this.orbitGoal) {
      // The camera moves with its orbit centre: the view glides to the new point.
      const step = this.scratch
        .subVectors(this.orbitGoal, this.controls.target)
        .multiplyScalar(1 - Math.exp(-seconds * 9));
      this.controls.target.add(step);
      this.camera.position.add(step);
      if (this.orbitGoal.distanceToSquared(this.controls.target) < 1e-4) this.orbitGoal = null;
    }
    this.controls.update();
  }

  /** Ends the damped motion of a drag in progress. */
  private settle(): void {
    this.controls.enableDamping = false;
    this.controls.update();
    this.controls.enableDamping = true;
  }

  dispose(): void {
    this.controls.dispose();
  }
}
