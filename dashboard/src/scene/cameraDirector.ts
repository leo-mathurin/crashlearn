/**
 * The cameras of the scene and the move from one to another. One camera is on screen at a
 * time: Aerial (orthographic, north up), Free (orbit) or Chase (behind the Voiture suivie).
 */
import * as THREE from "three/webgpu";

import type { Pose } from "../replay/playback";
import type { Bounds } from "../track/track";
import { AerialCamera, type Insets } from "./aerialCamera";
import { ChaseCamera } from "./chaseCamera";
import { FreeCamera } from "./freeCamera";

export type CameraMode = "aerial" | "free" | "chase";

export const CAMERA_MODES: readonly CameraMode[] = ["aerial", "free", "chase"];

export const CAMERA_LABELS: Record<CameraMode, string> = {
  aerial: "Aérienne",
  free: "Libre",
  chase: "Poursuite",
};

const TRANSITION_SECONDS = 0.65;
/**
 * A perspective camera with this field of view, far away, shows the ground almost as the
 * orthographic camera does: the move to or from the Aerial camera goes through it.
 */
const NEAR_ORTHOGRAPHIC_FOV_DEG = 2;

/** A camera placement that can be mixed with another one. */
interface View {
  /** Point on the ground that the camera looks at. */
  target: THREE.Vector3;
  orientation: THREE.Quaternion;
  /** Half of the view height at the target, in metres. */
  halfHeight: number;
  fovDeg: number;
}

function newView(): View {
  return {
    target: new THREE.Vector3(),
    orientation: new THREE.Quaternion(),
    halfHeight: 1,
    fovDeg: 45,
  };
}

function easeInOut(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
}

export class CameraDirector {
  readonly aerial: AerialCamera;
  readonly free: FreeCamera;
  readonly chase = new ChaseCamera();
  private current: CameraMode = "aerial";
  /** Camera on screen during a move between two cameras. */
  private readonly moving = new THREE.PerspectiveCamera();
  private transition: { from: View; elapsed: number } | null = null;
  private readonly to = newView();
  private readonly mixed = newView();
  private readonly forward = new THREE.Vector3();
  private viewportHeight = 1;
  private readonly reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  constructor(element: HTMLElement) {
    this.aerial = new AerialCamera(element);
    this.free = new FreeCamera(element);
    element.addEventListener(
      "wheel",
      (event) => {
        if (this.current !== "chase" || this.transition) return;
        event.preventDefault();
        this.chase.dolly(Math.exp(event.deltaY * 0.0015));
      },
      { passive: false },
    );
    element.addEventListener("dblclick", (event) => {
      if (this.transition) return;
      if (this.current === "aerial") this.aerial.reset();
      if (this.current === "free") this.free.orbitAround(event.clientX, event.clientY);
    });
  }

  get mode(): CameraMode {
    return this.current;
  }

  /** True during the move from one camera to another. */
  get inTransition(): boolean {
    return this.transition !== null;
  }

  /** Camera to render with at this frame. */
  get camera(): THREE.Camera {
    if (this.transition) return this.moving;
    return this.rigCamera(this.current);
  }

  /**
   * Changes the camera. With `animate`, the view glides from the old camera to the new one;
   * the playback position does not change.
   */
  setMode(mode: CameraMode, animate: boolean): void {
    if (mode === this.current) return;
    const from = newView();
    this.describe(this.current, from);
    if (this.transition) this.copyView(this.mixed, from);
    if (mode === "free" && !this.free.placed) {
      this.free.frame(from.target, Math.max(from.halfHeight, 3));
    }
    this.current = mode;
    this.aerial.enabled = false;
    this.free.enabled = false;
    if (animate && !this.reducedMotion.matches) {
      this.transition = { from, elapsed: 0 };
    } else {
      this.transition = null;
      this.enableControls();
    }
  }

  fit(bounds: Bounds, insets: Insets): void {
    this.aerial.fit(bounds, insets);
    this.free.placed = false;
    if (this.current === "free") this.free.frame(this.aerial.target, this.aerial.halfHeight);
  }

  /** Returns the current camera to its first view. */
  reset(): void {
    if (this.current === "aerial") this.aerial.reset();
    if (this.current === "free") this.free.frame(this.aerial.target, this.aerial.halfHeight);
  }

  resize(width: number, height: number): void {
    this.viewportHeight = height;
    this.aerial.resize(width, height);
    this.free.resize(width, height);
    this.chase.resize(width, height);
    this.moving.aspect = width / height;
  }

  /**
   * Advances the cameras by `seconds`. `followed` is the pose that the Chase camera follows, or
   * null when there is none. `snap` is true after a seek: the Chase camera does not glide.
   */
  update(seconds: number, followed: Pose | null, snap: boolean): void {
    this.aerial.update();
    this.free.update(seconds);
    if (followed) this.chase.update(seconds, followed, snap);

    const transition = this.transition;
    if (!transition) return;
    transition.elapsed += seconds;
    const progress = Math.min(transition.elapsed / TRANSITION_SECONDS, 1);
    // The goal is read again at each frame: the Chase camera moves with its car.
    this.describe(this.current, this.to);
    this.mix(transition.from, this.to, easeInOut(progress));
    this.apply(this.mixed);
    if (progress >= 1) {
      this.transition = null;
      this.enableControls();
    }
  }

  /** Scale of the scene at a point, in pixels per metre, for the size of the labels. */
  pixelsPerMetreAt(point: THREE.Vector3): number {
    const camera = this.camera;
    if (camera instanceof THREE.OrthographicCamera) return this.aerial.pixelsPerMetre;
    if (!(camera instanceof THREE.PerspectiveCamera)) return 1;
    const depth = this.forward
      .subVectors(point, camera.position)
      .dot(camera.getWorldDirection(new THREE.Vector3()));
    const visible = 2 * Math.max(depth, 0.01) * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    return this.viewportHeight / visible;
  }

  /** Centre and radius of the part of the ground that needs shadows. */
  shadowFocus(center: THREE.Vector3): number {
    if (this.transition) {
      center.copy(this.mixed.target);
      return this.mixed.halfHeight * 2.2;
    }
    if (this.current === "aerial") {
      center.copy(this.aerial.target);
      return this.aerial.viewRadius;
    }
    if (this.current === "free") {
      center.copy(this.free.target);
      return this.free.distance * 1.1;
    }
    center.copy(this.chase.target);
    return 28;
  }

  /** Distance from the camera to the point it looks at: the fog starts beyond it. */
  focusDistance(): number {
    if (this.transition) return this.moving.position.distanceTo(this.mixed.target);
    if (this.current === "aerial") return this.aerial.camera.position.y;
    if (this.current === "free") return this.free.distance;
    return this.chase.distance;
  }

  /** Centre of the view on the ground, in Three.js world axes. */
  focusPoint(out: THREE.Vector3): THREE.Vector3 {
    if (this.transition) return out.copy(this.mixed.target);
    if (this.current === "aerial") return out.copy(this.aerial.target);
    if (this.current === "free") return out.copy(this.free.target);
    return out.copy(this.chase.target);
  }

  private rigCamera(mode: CameraMode): THREE.Camera {
    if (mode === "aerial") return this.aerial.camera;
    return mode === "free" ? this.free.camera : this.chase.camera;
  }

  private enableControls(): void {
    this.aerial.enabled = this.current === "aerial";
    this.free.enabled = this.current === "free";
  }

  /** Writes the placement of the camera of `mode` into `view`. */
  private describe(mode: CameraMode, view: View): void {
    if (mode === "aerial") {
      const { camera } = this.aerial;
      view.target.copy(this.aerial.target);
      view.orientation.copy(camera.quaternion);
      view.halfHeight = this.aerial.halfHeight;
      view.fovDeg = NEAR_ORTHOGRAPHIC_FOV_DEG;
      return;
    }
    const camera = mode === "free" ? this.free.camera : this.chase.camera;
    const target = mode === "free" ? this.free.target : this.chase.target;
    view.target.copy(target);
    view.orientation.copy(camera.quaternion);
    view.fovDeg = camera.fov;
    view.halfHeight =
      camera.position.distanceTo(target) * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  }

  private copyView(from: View, to: View): void {
    to.target.copy(from.target);
    to.orientation.copy(from.orientation);
    to.halfHeight = from.halfHeight;
    to.fovDeg = from.fovDeg;
  }

  private mix(from: View, to: View, t: number): void {
    const { mixed } = this;
    mixed.target.lerpVectors(from.target, to.target, t);
    mixed.orientation.slerpQuaternions(from.orientation, to.orientation, t);
    // Sizes mix on a log scale: a zoom from 80 m to 1 m has the same pace all the way.
    mixed.halfHeight = Math.exp(
      THREE.MathUtils.lerp(Math.log(from.halfHeight), Math.log(to.halfHeight), t),
    );
    mixed.fovDeg = THREE.MathUtils.lerp(from.fovDeg, to.fovDeg, t);
  }

  /** Places the moving camera on a view. */
  private apply(view: View): void {
    const camera = this.moving;
    const distance = view.halfHeight / Math.tan(THREE.MathUtils.degToRad(view.fovDeg / 2));
    this.forward.set(0, 0, -1).applyQuaternion(view.orientation);
    camera.position.copy(view.target).addScaledVector(this.forward, -distance);
    camera.quaternion.copy(view.orientation);
    camera.fov = view.fovDeg;
    // The near-orthographic end is thousands of metres away: keep the depth range around the
    // ground, so that the depth buffer stays precise.
    camera.near = Math.max(0.05, distance - 400);
    camera.far = distance + 2500;
    camera.updateProjectionMatrix();
  }

  dispose(): void {
    this.aerial.dispose();
    this.free.dispose();
  }
}
