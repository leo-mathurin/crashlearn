/**
 * Optional scenery: sky, clouds, hills, trees, grandstands and the start gantry. It gives the
 * scene a setting and tells nothing about the race. It is drawn before the track, and the track
 * is then drawn over it (`Stage.render`): a tree can never hide a wall, a car or a LiDAR ray.
 *
 * Trees, grandstands and gantry are in simulator coordinates (x east, y north, z up). The sky
 * is in Three.js world axes.
 */
import * as THREE from "three/webgpu";

import {
  type DecorLayout,
  type Gantry,
  type Stand,
  STAND_DEPTH_M,
  type Tree,
} from "../track/decorLayout";
import type { Bounds } from "../track/track";
import { HAZE_COLOR, SCENERY_LAYER } from "./stage";

export interface Decor {
  /** Scenery in simulator coordinates: add it to the simulator root. */
  ground: THREE.Group;
  /** Sky, clouds and hills, in Three.js world axes: add it to the scene. */
  sky: THREE.Group;
  triangles: number;
  setVisible(visible: boolean): void;
  dispose(): void;
}

/** Deterministic generator: the clouds and the crowd are the same at each load. */
function random(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Sets one colour on every vertex of a geometry. */
function paint(geometry: THREE.BufferGeometry, color: number): THREE.BufferGeometry {
  const { r, g, b } = new THREE.Color(color);
  const count = geometry.getAttribute("position").count;
  const colors = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) colors.set([r, g, b], i * 3);
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return geometry;
}

/** Merges geometries that have position, normal and colour into one, without index. */
function merge(parts: THREE.BufferGeometry[]): THREE.BufferGeometry {
  const flat = parts.map((part) => (part.index ? part.toNonIndexed() : part));
  const merged = new THREE.BufferGeometry();
  for (const name of ["position", "normal", "color"]) {
    const arrays = flat.map((part) => part.getAttribute(name).array as Float32Array);
    const joined = new Float32Array(arrays.reduce((sum, array) => sum + array.length, 0));
    let offset = 0;
    for (const array of arrays) {
      joined.set(array, offset);
      offset += array.length;
    }
    merged.setAttribute(name, new THREE.BufferAttribute(joined, 3));
  }
  for (const part of [...parts, ...flat]) part.dispose();
  return merged;
}

/** A box in simulator axes: `size` and `center` are (x, y, z up). */
function box(
  size: readonly [number, number, number],
  center: readonly [number, number, number],
  color: number,
): THREE.BufferGeometry {
  return paint(new THREE.BoxGeometry(...size).translate(...center), color);
}

/** Sky dome: blue at the top, the haze colour at the horizon, so that the ground fades into it. */
function buildSky(): THREE.Mesh {
  const geometry = new THREE.SphereGeometry(2400, 32, 20);
  const position = geometry.getAttribute("position");
  const colors = new Float32Array(position.count * 3);
  const horizon = new THREE.Color(HAZE_COLOR);
  const middle = new THREE.Color(0x7fb0e2);
  const top = new THREE.Color(0x2f6fc4);
  const color = new THREE.Color();
  for (let i = 0; i < position.count; i++) {
    const height = Math.max(position.getY(i) / 2400, 0);
    if (height < 0.25) color.lerpColors(horizon, middle, height / 0.25);
    else color.lerpColors(middle, top, (height - 0.25) / 0.75);
    colors.set([color.r, color.g, color.b], i * 3);
  }
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return new THREE.Mesh(
    geometry,
    new THREE.MeshBasicMaterial({
      vertexColors: true,
      side: THREE.BackSide,
      fog: false,
      depthWrite: false,
    }),
  );
}

function buildClouds(next: () => number): THREE.InstancedMesh {
  const count = 34;
  const clouds = new THREE.InstancedMesh(
    new THREE.SphereGeometry(1, 12, 8),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.82, fog: false }),
    count * 3,
  );
  const matrix = new THREE.Matrix4();
  const rotation = new THREE.Quaternion();
  for (let cloud = 0; cloud < count; cloud++) {
    const angle = next() * Math.PI * 2;
    const distance = 500 + next() * 1300;
    const height = 260 + next() * 240;
    const size = 60 + next() * 90;
    // Three flat puffs side by side make one cloud.
    for (let puff = 0; puff < 3; puff++) {
      matrix.compose(
        new THREE.Vector3(
          Math.cos(angle) * distance + (puff - 1) * size * 0.75,
          height + (puff === 1 ? size * 0.08 : 0),
          Math.sin(angle) * distance + (next() - 0.5) * size * 0.4,
        ),
        rotation,
        new THREE.Vector3(size * (puff === 1 ? 1 : 0.7), size * 0.2, size * 0.6),
      );
      clouds.setMatrixAt(cloud * 3 + puff, matrix);
    }
  }
  return clouds;
}

/** A ring of low hills far away: a horizon line for the cameras near the ground. */
function buildHills(next: () => number): THREE.Mesh {
  const segments = 96;
  const radius = 760;
  const positions: number[] = [];
  const heights = Array.from({ length: segments }, () => 25 + next() * 70);
  for (let i = 0; i < segments; i++) {
    const j = (i + 1) % segments;
    const a = (i / segments) * Math.PI * 2;
    const b = (j / segments) * Math.PI * 2;
    // Smooth the crest: each summit is the mean of its neighbours.
    const top = (index: number): number =>
      ((heights[(index + segments - 1) % segments] ?? 0) +
        (heights[index] ?? 0) * 2 +
        (heights[(index + 1) % segments] ?? 0)) /
      4;
    const ax = Math.cos(a) * radius;
    const az = Math.sin(a) * radius;
    const bx = Math.cos(b) * radius;
    const bz = Math.sin(b) * radius;
    positions.push(
      ax,
      -2,
      az,
      bx,
      -2,
      bz,
      bx,
      top(j),
      bz,
      ax,
      -2,
      az,
      bx,
      top(j),
      bz,
      ax,
      top(i),
      az,
    );
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  return new THREE.Mesh(
    geometry,
    new THREE.MeshLambertMaterial({ color: 0x3d5a3a, side: THREE.DoubleSide }),
  );
}

/** Three.js builds cones along y: this turns one to the z axis of the simulator, at height z. */
function upright(geometry: THREE.BufferGeometry, z: number): THREE.BufferGeometry {
  return geometry.rotateX(Math.PI / 2).translate(0, 0, z);
}

/** One tree in simulator axes, about 1.3 m tall: the circuits are 1:10 scale models. */
function treeGeometry(): THREE.BufferGeometry {
  return merge([
    paint(upright(new THREE.CylinderGeometry(0.05, 0.075, 0.42, 7), 0.21), 0x5a4030),
    paint(upright(new THREE.ConeGeometry(0.5, 0.62, 9), 0.62), 0x2f6b34),
    paint(upright(new THREE.ConeGeometry(0.38, 0.52, 9), 0.94), 0x35783a),
    paint(upright(new THREE.ConeGeometry(0.24, 0.42, 9), 1.2), 0x3d8742),
  ]);
}

function buildTrees(trees: readonly Tree[]): THREE.InstancedMesh {
  const mesh = new THREE.InstancedMesh(
    treeGeometry(),
    new THREE.MeshLambertMaterial({ vertexColors: true }),
    trees.length,
  );
  const matrix = new THREE.Matrix4();
  const rotation = new THREE.Quaternion();
  const up = new THREE.Vector3(0, 0, 1);
  const color = new THREE.Color();
  trees.forEach((tree, index) => {
    rotation.setFromAxisAngle(up, tree.rotation);
    matrix.compose(
      new THREE.Vector3(tree.x, tree.y, 0),
      rotation,
      new THREE.Vector3(tree.scale, tree.scale, tree.scale * (0.9 + tree.tint * 0.3)),
    );
    mesh.setMatrixAt(index, matrix);
    // From a yellow green to a blue green.
    mesh.setColorAt(index, color.setHSL(0.27 + tree.tint * 0.1, 0.35, 0.72 + tree.tint * 0.2));
  });
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

const STAND_TIERS = 5;
const TIER_HEIGHT_M = 0.2;

/** Grandstand: stepped tiers, a back wall, a roof on posts. Its local +y side faces the track. */
function buildStand(stand: Stand, next: () => number): THREE.Group {
  const group = new THREE.Group();
  const { length } = stand;
  const tierDepth = STAND_DEPTH_M / STAND_TIERS;
  const parts: THREE.BufferGeometry[] = [];
  for (let tier = 0; tier < STAND_TIERS; tier++) {
    // The lowest tier is nearest to the track, at local +y.
    const y = STAND_DEPTH_M / 2 - tierDepth * (tier + 0.5);
    const height = TIER_HEIGHT_M * (tier + 1);
    parts.push(box([length, tierDepth, height], [0, y, height / 2], 0xb9bbb6));
  }
  const top = TIER_HEIGHT_M * STAND_TIERS;
  const roofHeight = top + 0.75;
  parts.push(
    box([length, 0.06, roofHeight], [0, -STAND_DEPTH_M / 2 - 0.03, roofHeight / 2], 0x8d9299),
  );
  parts.push(box([length + 0.3, STAND_DEPTH_M + 0.4, 0.05], [0, 0, roofHeight], 0x2b3a55));
  for (const side of [-1, 1]) {
    parts.push(
      box(
        [0.06, 0.06, roofHeight],
        [(side * length) / 2, STAND_DEPTH_M / 2, roofHeight / 2],
        0x8d9299,
      ),
    );
  }
  const structure = new THREE.Mesh(
    merge(parts),
    new THREE.MeshLambertMaterial({ vertexColors: true }),
  );
  structure.castShadow = true;
  structure.receiveShadow = true;

  // Spectators: one small coloured block per seat, some seats empty.
  const seats = Math.floor(length / 0.16);
  const crowd = new THREE.InstancedMesh(
    new THREE.BoxGeometry(0.09, 0.08, 0.15),
    new THREE.MeshLambertMaterial(),
    seats * STAND_TIERS,
  );
  const matrix = new THREE.Matrix4();
  const color = new THREE.Color();
  let placed = 0;
  for (let tier = 0; tier < STAND_TIERS; tier++) {
    for (let seat = 0; seat < seats; seat++) {
      if (next() < 0.22) continue;
      matrix.makeTranslation(
        -length / 2 + (seat + 0.5) * (length / seats),
        STAND_DEPTH_M / 2 - tierDepth * (tier + 0.5),
        TIER_HEIGHT_M * (tier + 1) + 0.075,
      );
      crowd.setMatrixAt(placed, matrix);
      crowd.setColorAt(placed, color.setHSL(next(), 0.55, 0.45 + next() * 0.25));
      placed++;
    }
  }
  crowd.count = placed;
  group.add(structure, crowd);
  group.position.set(stand.x, stand.y, 0);
  // Local +y is the left of the heading: turn half a turn when the track is on the right.
  group.rotation.z = stand.heading + (stand.facing === 1 ? 0 : Math.PI);
  return group;
}

/** Banner of the gantry, drawn on a canvas. */
function bannerTexture(): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = 1024;
  canvas.height = 96;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("2D canvas is not available");
  context.fillStyle = "#14171d";
  context.fillRect(0, 0, canvas.width, canvas.height);
  const square = 24;
  for (const origin of [0, canvas.width - square * 4]) {
    for (let column = 0; column < 4; column++) {
      for (let row = 0; row < 4; row++) {
        context.fillStyle = (column + row) % 2 === 0 ? "#f2f2ee" : "#14171d";
        context.fillRect(origin + column * square, row * square, square, square);
      }
    }
  }
  context.fillStyle = "#ffd23f";
  context.font = "700 58px system-ui, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText("CRASH & LEARN", canvas.width / 2, canvas.height / 2 + 3);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

const GANTRY_HEIGHT_M = 1.25;

/**
 * Start gantry: a post on each side, outside the DNF limit, and a beam above the track with a
 * banner. It casts no shadow: nothing of it reaches the track surface.
 */
function buildGantry(gantry: Gantry): THREE.Group {
  const group = new THREE.Group();
  const dx = gantry.right.x - gantry.left.x;
  const dy = gantry.right.y - gantry.left.y;
  const span = Math.hypot(dx, dy);
  const parts = [
    box([0.1, 0.1, GANTRY_HEIGHT_M], [-span / 2, 0, GANTRY_HEIGHT_M / 2], 0x9aa0a8),
    box([0.1, 0.1, GANTRY_HEIGHT_M], [span / 2, 0, GANTRY_HEIGHT_M / 2], 0x9aa0a8),
    box([span + 0.1, 0.1, 0.06], [0, 0, GANTRY_HEIGHT_M - 0.28], 0x9aa0a8),
  ];
  const frame = new THREE.Mesh(merge(parts), new THREE.MeshLambertMaterial({ vertexColors: true }));
  const bannerHeight = 0.22;
  const banner = new THREE.Mesh(
    new THREE.BoxGeometry(span + 0.1, 0.05, bannerHeight),
    // Not lit: the banner reads the same from both sides, in the sun or not.
    new THREE.MeshBasicMaterial({ map: bannerTexture() }),
  );
  banner.position.set(0, 0, GANTRY_HEIGHT_M - bannerHeight / 2);
  group.add(frame, banner);
  group.position.set((gantry.left.x + gantry.right.x) / 2, (gantry.left.y + gantry.right.y) / 2, 0);
  group.rotation.z = Math.atan2(dy, dx);
  return group;
}

function triangleCount(object: THREE.Object3D): number {
  let total = 0;
  object.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    const geometry: THREE.BufferGeometry = child.geometry;
    const triangles = (geometry.index?.count ?? geometry.getAttribute("position").count) / 3;
    total += triangles * (child instanceof THREE.InstancedMesh ? child.count : 1);
  });
  return total;
}

/**
 * Builds the scenery. `layout` is null when the map of the replay is not available: there is
 * then a sky and hills, and nothing near the cars. `bounds` centres the sky on the circuit.
 */
export function buildDecor(layout: DecorLayout | null, bounds: Bounds, seed: number): Decor {
  const next = random(seed);
  const sky = new THREE.Group();
  sky.name = "decor:sky";
  sky.add(buildSky(), buildClouds(next), buildHills(next));
  // Simulator (x, y) is (x, -z) in Three.js world axes.
  sky.position.set((bounds.minX + bounds.maxX) / 2, 0, -(bounds.minY + bounds.maxY) / 2);

  const ground = new THREE.Group();
  ground.name = "decor:ground";
  if (layout) {
    if (layout.trees.length > 0) ground.add(buildTrees(layout.trees));
    for (const stand of layout.stands) ground.add(buildStand(stand, next));
    if (layout.gantry) ground.add(buildGantry(layout.gantry));
  }
  for (const root of [sky, ground]) {
    root.traverse((child) => child.layers.set(SCENERY_LAYER));
  }
  return {
    ground,
    sky,
    triangles: triangleCount(sky) + triangleCount(ground),
    setVisible(visible) {
      sky.visible = visible;
      ground.visible = visible;
    },
    dispose() {
      for (const root of [sky, ground]) {
        root.traverse((child) => {
          if (!(child instanceof THREE.Mesh)) return;
          child.geometry.dispose();
          const material: THREE.Material & { map?: THREE.Texture | null } = child.material;
          material.map?.dispose();
          material.dispose();
        });
      }
    },
  };
}
