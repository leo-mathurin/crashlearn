/**
 * Screen markers of the cars. A car is 0.58 m long on a circuit of several hundred metres:
 * seen from far, it is smaller than a pixel. A ring of constant screen size shows where it is.
 */
import type * as THREE from "three/webgpu";

import { carColor } from "../scene/cars";
import { h } from "./dom";

/** The ring fades out when the car itself is longer than this on screen, in pixels. */
const CAR_VISIBLE_PX = 34;

export class CarMarkers {
  private readonly markers: HTMLElement[] = [];

  constructor(private readonly layer: HTMLElement) {}

  setCars(count: number): void {
    this.layer.replaceChildren();
    this.markers.length = 0;
    for (let car = 0; car < count; car++) {
      const marker = h("div", { class: "car-marker" });
      marker.style.setProperty("--car", carColor(car));
      this.markers.push(marker);
      this.layer.append(marker);
    }
  }

  /**
   * Moves the marker of a car. `position` is in Three.js world axes and is changed.
   * `carPixels` is the length of the car on screen.
   */
  place(
    car: number,
    visible: boolean,
    position: THREE.Vector3,
    camera: THREE.Camera,
    width: number,
    height: number,
    carPixels: number,
  ): void {
    const marker = this.markers[car];
    if (!marker) return;
    position.project(camera);
    const onScreen =
      visible && position.z < 1 && Math.abs(position.x) <= 1.05 && Math.abs(position.y) <= 1.05;
    if (!onScreen || carPixels > CAR_VISIBLE_PX) {
      marker.style.opacity = "0";
      return;
    }
    const x = (position.x * 0.5 + 0.5) * width;
    const y = (-position.y * 0.5 + 0.5) * height;
    marker.style.opacity = "1";
    marker.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0)`;
  }
}
