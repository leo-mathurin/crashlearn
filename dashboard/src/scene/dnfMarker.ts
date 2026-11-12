/**
 * Ground mark at the last pose of a car that ended in DNF: a red ring with a cross, in
 * simulator coordinates. The "DNF" tag itself is a screen label (`ui/labels.ts`).
 */
import * as THREE from "three/webgpu";

const RADIUS_M = 0.42;
const Z = 0.02;

const material = new THREE.MeshBasicMaterial({
  color: 0xff5a47,
  transparent: true,
  opacity: 0.92,
  depthWrite: false,
  side: THREE.DoubleSide,
  polygonOffset: true,
  polygonOffsetFactor: -6,
  polygonOffsetUnits: -6,
});

export function buildDnfMarker(x: number, y: number): THREE.Group {
  const group = new THREE.Group();
  const ring = new THREE.Mesh(new THREE.RingGeometry(RADIUS_M * 0.84, RADIUS_M, 40), material);
  group.add(ring);
  for (const angle of [Math.PI / 4, -Math.PI / 4]) {
    const bar = new THREE.Mesh(new THREE.PlaneGeometry(RADIUS_M * 1.25, RADIUS_M * 0.16), material);
    bar.rotation.z = angle;
    group.add(bar);
  }
  group.position.set(x, y, Z);
  group.name = "dnf-marker";
  group.visible = false;
  return group;
}
