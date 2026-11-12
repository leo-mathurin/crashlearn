/**
 * Screen labels anchored to the scene: the name of each car, a dot where a car is too small to
 * see, and the "DNF" tag at the last pose of a car that ended in DNF.
 */
import type * as THREE from "three/webgpu";

import { carColor } from "../scene/cars";
import { h } from "./dom";

/** The dot fades out when the car itself is longer than this on screen, in pixels. */
const CAR_VISIBLE_PX = 34;

export class SceneLabels {
  private readonly cars: HTMLElement[] = [];
  private readonly dnf: HTMLElement[] = [];

  constructor(
    private readonly layer: HTMLElement,
    private readonly onSelect: (car: number) => void,
  ) {}

  setCars(names: readonly string[]): void {
    this.layer.replaceChildren();
    this.cars.length = 0;
    this.dnf.length = 0;
    names.forEach((name, car) => {
      const label = h(
        "button",
        {
          class: "car-label",
          type: "button",
          hidden: true,
          tabindex: -1,
          onclick: () => this.onSelect(car),
        },
        h("span", { class: "car-dot" }),
        h("span", { class: "car-tag" }, name),
      );
      const tag = h("div", { class: "dnf-tag", hidden: true }, h("strong", {}, "DNF"), name);
      for (const element of [label, tag]) element.style.setProperty("--car", carColor(car));
      this.cars.push(label);
      this.dnf.push(tag);
      this.layer.append(tag, label);
    });
  }

  setSelected(selected: number): void {
    this.cars.forEach((label, car) => label.toggleAttribute("data-selected", car === selected));
  }

  /**
   * Moves the label of a car. `position` is in Three.js world axes and is changed.
   * `carPixels` is the length of the car on screen.
   */
  placeCar(
    car: number,
    visible: boolean,
    position: THREE.Vector3,
    camera: THREE.Camera,
    width: number,
    height: number,
    carPixels: number,
  ): void {
    const label = this.cars[car];
    if (!label) return;
    if (!place(label, visible, position, camera, width, height)) return;
    label.toggleAttribute("data-small", carPixels <= CAR_VISIBLE_PX);
    // The tag sits above the car, whatever its size on screen.
    label.style.setProperty("--lift", `${Math.max(carPixels * 0.5, 9).toFixed(0)}px`);
  }

  placeDnf(
    car: number,
    visible: boolean,
    position: THREE.Vector3,
    camera: THREE.Camera,
    width: number,
    height: number,
  ): void {
    const tag = this.dnf[car];
    if (tag) place(tag, visible, position, camera, width, height);
  }
}

/** Projects `position` and moves `element` there. Returns false when it is not on screen. */
function place(
  element: HTMLElement,
  visible: boolean,
  position: THREE.Vector3,
  camera: THREE.Camera,
  width: number,
  height: number,
): boolean {
  position.project(camera);
  const onScreen =
    visible && position.z < 1 && Math.abs(position.x) <= 1.05 && Math.abs(position.y) <= 1.05;
  if (!onScreen) {
    if (!element.hidden) element.hidden = true;
    return false;
  }
  if (element.hidden) element.hidden = false;
  const x = (position.x * 0.5 + 0.5) * width;
  const y = (-position.y * 0.5 + 0.5) * height;
  element.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0)`;
  return true;
}
