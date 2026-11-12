/**
 * Car models, in simulator coordinates: x forward, y to the left, z up. The footprint is the
 * collision box of the simulator, 0.58 m long and 0.31 m wide, centred on the recorded pose.
 */
import * as THREE from "three/webgpu";

export const CAR_LENGTH_M = 0.58;
export const CAR_WIDTH_M = 0.31;

/** One colour per grid slot, the same as in `vendor/simulation/viz_replay.py`. */
export const CAR_COLORS = ["#ff6b35", "#4cc9f0", "#7bf178", "#f72585"] as const;

export function carColor(car: number): string {
  return CAR_COLORS[car % CAR_COLORS.length] ?? CAR_COLORS[0];
}

const HALF_LENGTH = CAR_LENGTH_M / 2;
const HALF_WIDTH = CAR_WIDTH_M / 2;
const WHEEL_RADIUS = 0.052;
const WHEEL_WIDTH = 0.05;
const WHEEL_Y = HALF_WIDTH - WHEEL_WIDTH / 2 - 0.002;
const FRONT_AXLE_X = 0.185;
const REAR_AXLE_X = -0.19;

/** Top outline of the body, one side: nose, cockpit flanks, sidepods, tail. */
const BODY_OUTLINE: readonly (readonly [number, number])[] = [
  [HALF_LENGTH - 0.02, 0.022],
  [0.2, 0.04],
  [0.09, 0.062],
  [0.03, 0.098],
  [-0.115, 0.098],
  [-0.2, 0.062],
  [-0.25, 0.05],
];

const shared = {
  carbon: new THREE.MeshStandardMaterial({ color: 0x17181c, roughness: 0.45, metalness: 0.3 }),
  rubber: new THREE.MeshStandardMaterial({ color: 0x0b0b0c, roughness: 0.92 }),
  rim: new THREE.MeshStandardMaterial({ color: 0x9a9da3, roughness: 0.35, metalness: 0.8 }),
  glass: new THREE.MeshPhysicalMaterial({
    color: 0x0a0d12,
    roughness: 0.08,
    metalness: 0.2,
    clearcoat: 1,
  }),
  lamp: new THREE.MeshStandardMaterial({
    color: 0xfff6dd,
    emissive: 0xfff0c2,
    emissiveIntensity: 1.6,
  }),
};

function bodyGeometry(): THREE.BufferGeometry {
  const shape = new THREE.Shape();
  const first = BODY_OUTLINE[0] ?? [0, 0];
  shape.moveTo(first[0], first[1]);
  for (const [x, y] of BODY_OUTLINE.slice(1)) shape.lineTo(x, y);
  for (const [x, y] of BODY_OUTLINE.toReversed()) shape.lineTo(x, -y);
  shape.closePath();
  return new THREE.ExtrudeGeometry(shape, {
    depth: 0.05,
    bevelEnabled: true,
    bevelThickness: 0.012,
    bevelSize: 0.012,
    bevelSegments: 3,
  });
}

function box(
  size: readonly [number, number, number],
  position: readonly [number, number, number],
  material: THREE.Material,
): THREE.Mesh {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), material);
  mesh.position.set(...position);
  return mesh;
}

function wheel(): THREE.Group {
  const group = new THREE.Group();
  // A cylinder has its axis on y: the lateral axis of the car.
  const tyre = new THREE.Mesh(
    new THREE.CylinderGeometry(WHEEL_RADIUS, WHEEL_RADIUS, WHEEL_WIDTH, 24),
    shared.rubber,
  );
  const rim = new THREE.Mesh(
    new THREE.CylinderGeometry(WHEEL_RADIUS * 0.58, WHEEL_RADIUS * 0.58, WHEEL_WIDTH + 0.004, 16),
    shared.rim,
  );
  group.add(tyre, rim);
  return group;
}

export class CarView {
  readonly group = new THREE.Group();
  private readonly frontWheels: THREE.Group[] = [];

  constructor(readonly car: number) {
    const paint = new THREE.MeshPhysicalMaterial({
      color: carColor(car),
      roughness: 0.42,
      metalness: 0.05,
      clearcoat: 0.5,
      clearcoatRoughness: 0.2,
      envMapIntensity: 0.6,
    });
    const body = new THREE.Mesh(bodyGeometry(), paint);
    body.position.z = 0.04;

    const cockpit = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 12), shared.glass);
    cockpit.scale.set(0.075, 0.042, 0.034);
    cockpit.position.set(-0.01, 0, 0.102);

    const engineCover = box([0.15, 0.05, 0.035], [-0.13, 0, 0.115], paint);
    const frontWing = box(
      [0.05, CAR_WIDTH_M - 0.01, 0.012],
      [HALF_LENGTH - 0.027, 0, 0.03],
      shared.carbon,
    );
    const rearWing = box([0.055, 0.26, 0.012], [-HALF_LENGTH + 0.03, 0, 0.14], shared.carbon);
    const plates = [-1, 1].map((side) =>
      box([0.07, 0.008, 0.08], [-HALF_LENGTH + 0.036, side * 0.13, 0.108], paint),
    );
    const lamps = [-1, 1].map((side) =>
      box([0.008, 0.02, 0.012], [0.203, side * 0.03, 0.078], shared.lamp),
    );
    this.group.add(body, cockpit, engineCover, frontWing, rearWing, ...plates, ...lamps);

    for (const [x, steers] of [
      [FRONT_AXLE_X, true],
      [REAR_AXLE_X, false],
    ] as const) {
      for (const side of [-1, 1]) {
        const group = wheel();
        group.position.set(x, side * WHEEL_Y, WHEEL_RADIUS);
        if (steers) this.frontWheels.push(group);
        this.group.add(group);
      }
    }
    this.group.traverse((child) => {
      child.castShadow = true;
      child.receiveShadow = true;
    });
    this.group.name = `car:${car}`;
  }

  /** Places the car at a recorded pose: x, y in metres, heading in radians. */
  setPose(x: number, y: number, heading: number): void {
    this.group.position.set(x, y, 0);
    this.group.rotation.z = heading;
  }

  /** Turns the front wheels by the recorded steering angle, in radians. */
  setSteering(angle: number): void {
    for (const wheelGroup of this.frontWheels) wheelGroup.rotation.z = angle;
  }

  dispose(): void {
    this.group.traverse((child) => {
      if (child instanceof THREE.Mesh) child.geometry.dispose();
    });
  }
}
