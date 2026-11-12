/**
 * Aerial camera: orthographic, straight down, north at the top of the screen. Zoom with the
 * wheel or a pinch, move by dragging.
 */
import { MapControls } from "three/addons/controls/MapControls.js";
import * as THREE from "three/webgpu";

import type { Bounds } from "../track/track";

const HEIGHT_M = 200;
/** The closest zoom shows this many metres from the top to the bottom of the screen. */
const MIN_VIEW_HEIGHT_M = 2.5;

export interface Insets {
  top: number;
  bottom: number;
  left: number;
  right: number;
}

const NO_INSETS: Insets = { top: 0, bottom: 0, left: 0, right: 0 };

export class AerialCamera {
  readonly camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 1, HEIGHT_M * 2);
  private readonly controls: MapControls;
  private aspect = 1;
  /** Half of the view height in metres at zoom 1: the whole circuit with a margin. */
  private fitHalfHeight = 50;
  private readonly home = new THREE.Vector3();

  constructor(element: HTMLElement) {
    this.controls = new MapControls(this.camera, element);
    this.controls.enableRotate = false;
    this.controls.screenSpacePanning = true;
    this.controls.zoomToCursor = true;
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.18;
    this.controls.zoomSpeed = 1.4;
    // Straight down. With the camera offset on +z, the top of the screen is -z: north.
    this.controls.minPolarAngle = 0;
    this.controls.maxPolarAngle = 0;
    this.controls.minAzimuthAngle = 0;
    this.controls.maxAzimuthAngle = 0;
    this.controls.touches = { ONE: THREE.TOUCH.PAN, TWO: THREE.TOUCH.DOLLY_PAN };
  }

  set enabled(value: boolean) {
    this.controls.enabled = value;
  }

  /** Centre of the view on the ground, in Three.js world axes. */
  get target(): THREE.Vector3 {
    return this.controls.target;
  }

  /** Half of the diagonal of the view on the ground, in metres. */
  get viewRadius(): number {
    const halfHeight = this.fitHalfHeight / this.camera.zoom;
    return Math.hypot(halfHeight, halfHeight * this.aspect);
  }

  get pixelsPerMetre(): number {
    return this.viewportHeight / ((this.fitHalfHeight * 2) / this.camera.zoom);
  }

  private viewportHeight = 1;

  /**
   * Frames an area given in simulator metres, and makes it the view that `reset` returns to.
   * `insets` are the pixels that the HUD covers on each side of the screen.
   */
  fit(bounds: Bounds, insets: Insets = NO_INSETS): void {
    const width = bounds.maxX - bounds.minX;
    const height = bounds.maxY - bounds.minY;
    const margin = 1.06;
    const viewportWidth = this.viewportHeight * this.aspect;
    const freeHeight = Math.max(this.viewportHeight - insets.top - insets.bottom, 100);
    const freeWidth = Math.max(viewportWidth - insets.left - insets.right, 100);
    const metresPerPixel = Math.max(height / freeHeight, width / freeWidth) * margin;
    this.fitHalfHeight = (metresPerPixel * this.viewportHeight) / 2;
    // The free area is not centred on the screen: move the view centre by the difference.
    const centerX =
      (bounds.minX + bounds.maxX) / 2 + ((insets.right - insets.left) / 2) * metresPerPixel;
    const centerY =
      (bounds.minY + bounds.maxY) / 2 - ((insets.bottom - insets.top) / 2) * metresPerPixel;
    this.home.set(centerX, 0, -centerY);
    this.controls.minZoom = 0.6;
    this.controls.maxZoom = (this.fitHalfHeight * 2) / MIN_VIEW_HEIGHT_M;
    this.reset();
  }

  /** Centres the view on a point in simulator metres, with `viewHeight` metres on screen. */
  focus(x: number, y: number, viewHeight: number): void {
    this.settle();
    this.camera.zoom = (this.fitHalfHeight * 2) / viewHeight;
    this.controls.target.set(x, 0, -y);
    this.camera.position.set(x, HEIGHT_M, -y);
    this.applyFrustum();
    this.controls.update();
  }

  reset(): void {
    this.settle();
    this.camera.zoom = 1;
    this.controls.target.copy(this.home);
    this.camera.position.set(this.home.x, HEIGHT_M, this.home.z);
    this.applyFrustum();
    this.controls.update();
  }

  resize(width: number, height: number): void {
    this.aspect = width / height;
    this.viewportHeight = height;
    this.applyFrustum();
  }

  update(): void {
    this.controls.update();
  }

  /** Ends the damped motion of a drag in progress, so that it does not move the next view. */
  private settle(): void {
    this.controls.enableDamping = false;
    this.controls.update();
    this.controls.enableDamping = true;
  }

  private applyFrustum(): void {
    this.camera.top = this.fitHalfHeight;
    this.camera.bottom = -this.fitHalfHeight;
    this.camera.left = -this.fitHalfHeight * this.aspect;
    this.camera.right = this.fitHalfHeight * this.aspect;
    this.camera.updateProjectionMatrix();
  }

  dispose(): void {
    this.controls.dispose();
  }
}
