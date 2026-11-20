/**
 * Mini-map: an aerial view of the whole circuit, always on screen, to see where the cars are.
 * It is not a camera and it takes no click. The circuit is a picture of the 3D scene, taken
 * from above with the rendering of the Aerial camera; the cars and DNF marks are drawn on it.
 */
import * as THREE from "three/webgpu";

import type { Bounds } from "../track/track";
import { h } from "./dom";

const SIZE_PX = 208;
/** Share of the map side that the circuit fills. */
const FILL = 0.94;

export interface MiniMapCar {
  x: number;
  y: number;
  color: string;
  selected: boolean;
}

export class MiniMap {
  readonly element: HTMLElement;
  private readonly canvas = h("canvas", { class: "minimap-canvas" });
  /** Picture of the circuit, without cars. */
  private readonly base = document.createElement("canvas");
  /** Camera that takes the picture: orthographic, straight down, north up. */
  readonly camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 1, 400);
  /** True when the picture of the circuit must be taken again. */
  stale = true;
  private centerX = 0;
  private centerY = 0;
  /** Half of the side of the square area on the map, in metres. */
  private halfSide = 50;
  private readonly ratio = Math.min(window.devicePixelRatio, 2);

  constructor() {
    this.element = h(
      "div",
      { class: "panel minimap", hidden: true, role: "img", "aria-label": "Mini-carte du circuit" },
      this.canvas,
    );
    for (const canvas of [this.canvas, this.base]) {
      canvas.width = SIZE_PX * this.ratio;
      canvas.height = SIZE_PX * this.ratio;
    }
    this.camera.up.set(0, 0, -1);
  }

  /** Sets the area that the map shows, in simulator metres. */
  setBounds(bounds: Bounds): void {
    this.centerX = (bounds.minX + bounds.maxX) / 2;
    this.centerY = (bounds.minY + bounds.maxY) / 2;
    this.halfSide = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY) / 2 / FILL;
    this.camera.position.set(this.centerX, 200, -this.centerY);
    this.camera.lookAt(this.centerX, 0, -this.centerY);
    this.stale = true;
    this.element.hidden = false;
  }

  /** Radius of the area of the map, for the shadows of the picture. */
  get radius(): number {
    return this.halfSide * Math.SQRT2;
  }

  /** Centre of the map in Three.js world axes. */
  center(out: THREE.Vector3): THREE.Vector3 {
    return out.set(this.centerX, 0, -this.centerY);
  }

  /** Sets the camera so that the square area is in the middle of a view of this size. */
  prepare(width: number, height: number): void {
    const aspect = width / height;
    const halfHeight = aspect >= 1 ? this.halfSide : this.halfSide / aspect;
    this.camera.top = halfHeight;
    this.camera.bottom = -halfHeight;
    this.camera.left = -halfHeight * aspect;
    this.camera.right = halfHeight * aspect;
    this.camera.updateProjectionMatrix();
    this.camera.updateMatrixWorld();
  }

  /** Keeps the middle square of a frame that was just rendered with `camera`. */
  capture(source: HTMLCanvasElement): void {
    const context = this.base.getContext("2d");
    if (!context) return;
    const side = Math.min(source.width, source.height);
    context.drawImage(
      source,
      (source.width - side) / 2,
      (source.height - side) / 2,
      side,
      side,
      0,
      0,
      this.base.width,
      this.base.height,
    );
    this.stale = false;
  }

  /**
   * Draws the map: the circuit, a red cross at each DNF mark, a dot for each active car, and
   * the place that the camera looks at.
   */
  draw(
    cars: readonly (MiniMapCar | null)[],
    dnf: readonly { x: number; y: number }[],
    focus: { x: number; y: number },
  ): void {
    const context = this.canvas.getContext("2d");
    if (!context) return;
    const scale = (SIZE_PX * this.ratio) / (this.halfSide * 2);
    const u = (x: number): number => (x - this.centerX + this.halfSide) * scale;
    const v = (y: number): number => (this.centerY + this.halfSide - y) * scale;
    const unit = this.ratio;
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.drawImage(this.base, 0, 0);

    // Place that the camera looks at.
    context.strokeStyle = "rgb(255 255 255 / 0.75)";
    context.lineWidth = 1.25 * unit;
    context.beginPath();
    context.arc(u(focus.x), v(focus.y), 9 * unit, 0, Math.PI * 2);
    context.stroke();

    context.lineCap = "round";
    for (const mark of dnf) {
      const size = 4 * unit;
      context.strokeStyle = "#14161b";
      context.lineWidth = 4.5 * unit;
      for (const pass of [0, 1]) {
        context.beginPath();
        context.moveTo(u(mark.x) - size, v(mark.y) - size);
        context.lineTo(u(mark.x) + size, v(mark.y) + size);
        context.moveTo(u(mark.x) + size, v(mark.y) - size);
        context.lineTo(u(mark.x) - size, v(mark.y) + size);
        context.stroke();
        if (pass === 0) {
          context.strokeStyle = "#ff5a47";
          context.lineWidth = 2.25 * unit;
        }
      }
    }
    // The selected car is drawn last, on top of the others.
    for (const selected of [false, true]) {
      for (const car of cars) {
        if (!car || car.selected !== selected) continue;
        context.beginPath();
        context.arc(u(car.x), v(car.y), (selected ? 5 : 4) * unit, 0, Math.PI * 2);
        context.fillStyle = car.color;
        context.fill();
        context.lineWidth = (selected ? 2 : 1.5) * unit;
        context.strokeStyle = selected ? "#ffffff" : "#14161b";
        context.stroke();
      }
    }
  }
}
