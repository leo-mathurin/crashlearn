/**
 * Free camera: a perspective camera in orbit around a point on the ground. Orbit, zoom and
 * move with the mouse, the trackpad or touch; a double-click moves the orbit point.
 */
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import * as THREE from "three/webgpu";

const FOV_DEG = 42;
/** Polar angle of the first view, from the vertical: a three-quarter view from the south. */
const FIRST_POLAR_RAD = (52 * Math.PI) / 180;

export class FreeCamera {
  readonly camera = new THREE.PerspectiveCamera(FOV_DEG, 1, 0.1, 4000);
  private readonly controls: OrbitControls;
  private readonly raycaster = new THREE.Raycaster();
  private readonly ground = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  /** Point that the orbit centre moves to after a double-click, or null. */
  private orbitGoal: THREE.Vector3 | null = null;
  private readonly scratch = new THREE.Vector3();
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

  update(seconds: number): void {
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
