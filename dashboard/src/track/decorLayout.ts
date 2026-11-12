/**
 * Where the optional scenery stands: trees, grandstands and the start gantry. Every item is
 * outside the track corridor and outside the DNF limit, by a clearance checked with the
 * off-track rule of the simulator. Positions are simulator metres. No rendering code here.
 */
import {
  type Centerline,
  curvature,
  distancePastEdge,
  OFFTRACK_MARGIN_M,
  startPose,
} from "./track";

/** Smallest distance between the DNF limit and any part of a tree or a grandstand, in metres. */
export const DECOR_CLEARANCE_M = 4;
/** The posts of the start gantry stand this far outside the DNF limit. */
export const GANTRY_CLEARANCE_M = 0.4;

const TREE_GRID_M = 3.2;
const TREE_RADIUS_M = 0.65;
const TREE_AREA_MARGIN_M = 55;
const MAX_TREES = 900;

export const STAND_DEPTH_M = 2.2;
const STAND_MAX_LENGTH_M = 14;
const STAND_MIN_STRAIGHT_M = 20;
const MAX_STANDS = 3;
/** The centerline is straight where it turns less than this, in rad/m. */
const STRAIGHT_CURVATURE_RADPM = 0.03;

export interface Tree {
  x: number;
  y: number;
  /** Size factor around 1. */
  scale: number;
  /** Rotation around the vertical axis, in radians. */
  rotation: number;
  /** Colour variation, from 0 to 1. */
  tint: number;
}

export interface Stand {
  /** Centre of the footprint. */
  x: number;
  y: number;
  /** Direction of the long side, in radians: parallel to the track. */
  heading: number;
  length: number;
  /** 1 when the track is on the left of `heading`, -1 when it is on the right. */
  facing: 1 | -1;
}

export interface Gantry {
  /** Feet of the two posts. */
  left: { x: number; y: number };
  right: { x: number; y: number };
}

export interface DecorLayout {
  trees: Tree[];
  stands: Stand[];
  gantry: Gantry | null;
}

/** Deterministic number in [0, 1) from two integers and a seed: the same scenery at each load. */
function hash(a: number, b: number, seed: number): number {
  let h = Math.imul(a, 0x27d4eb2d) ^ Math.imul(b, 0x165667b1) ^ Math.imul(seed, 0x9e3779b1);
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function seedOf(text: string): number {
  let seed = 2166136261;
  for (const char of text) seed = Math.imul(seed ^ (char.codePointAt(0) ?? 0), 16777619);
  return seed >>> 0;
}

/** Smooth noise in [0, 1]: trees grow in groves, not on a regular grid. */
function groveNoise(x: number, y: number, seed: number): number {
  const cell = 26;
  const gx = Math.floor(x / cell);
  const gy = Math.floor(y / cell);
  const fx = x / cell - gx;
  const fy = y / cell - gy;
  const sx = fx * fx * (3 - 2 * fx);
  const sy = fy * fy * (3 - 2 * fy);
  const top = hash(gx, gy, seed) * (1 - sx) + hash(gx + 1, gy, seed) * sx;
  const bottom = hash(gx, gy + 1, seed) * (1 - sx) + hash(gx + 1, gy + 1, seed) * sx;
  return top * (1 - sy) + bottom * sy;
}

/** Corners of a grandstand footprint, as x, y pairs. */
export function standCorners(stand: Stand): number[] {
  const cos = Math.cos(stand.heading);
  const sin = Math.sin(stand.heading);
  const corners: number[] = [];
  for (const [along, across] of [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
  ] as const) {
    const u = (along * stand.length) / 2;
    const v = (across * STAND_DEPTH_M) / 2;
    corners.push(stand.x + cos * u - sin * v, stand.y + sin * u + cos * v);
  }
  return corners;
}

function standIsClear(line: Centerline, stand: Stand): boolean {
  const corners = standCorners(stand);
  // Corners, and points along the long sides: a turn of the track can come close in between.
  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4;
    for (let step = 0; step <= 6; step++) {
      const t = step / 6;
      const x = (corners[i * 2] ?? 0) * (1 - t) + (corners[j * 2] ?? 0) * t;
      const y = (corners[i * 2 + 1] ?? 0) * (1 - t) + (corners[j * 2 + 1] ?? 0) * t;
      if (distancePastEdge(line, x, y, OFFTRACK_MARGIN_M) < DECOR_CLEARANCE_M) return false;
    }
  }
  return true;
}

function layoutStands(line: Centerline): Stand[] {
  const bend = curvature(line);
  const count = bend.length;
  // Runs of straight waypoints, longest first.
  const runs: { from: number; to: number; length: number }[] = [];
  let from = -1;
  for (let i = 0; i <= count; i++) {
    const straight = i < count && (bend[i] ?? 0) < STRAIGHT_CURVATURE_RADPM;
    if (straight && from < 0) from = i;
    if (!straight && from >= 0) {
      const length = (line.arc[i - 1] ?? 0) - (line.arc[from] ?? 0);
      if (length >= STAND_MIN_STRAIGHT_M) runs.push({ from, to: i - 1, length });
      from = -1;
    }
  }
  runs.sort((a, b) => b.length - a.length);

  const stands: Stand[] = [];
  for (const run of runs) {
    if (stands.length >= MAX_STANDS) break;
    const middle = Math.floor((run.from + run.to) / 2);
    const tx = (line.x[middle + 1] ?? 0) - (line.x[middle] ?? 0);
    const ty = (line.y[middle + 1] ?? 0) - (line.y[middle] ?? 0);
    const norm = Math.hypot(tx, ty) || 1;
    const heading = Math.atan2(ty, tx);
    const length = Math.min(run.length * 0.45, STAND_MAX_LENGTH_M);
    for (const side of [1, -1] as const) {
      // side 1: the stand is on the right of the direction of travel, the track on its left.
      const width = (side === 1 ? line.right[middle] : line.left[middle]) ?? 0;
      const offset = width + OFFTRACK_MARGIN_M + DECOR_CLEARANCE_M + STAND_DEPTH_M / 2 + 0.3;
      const stand: Stand = {
        x: (line.x[middle] ?? 0) + ((side * ty) / norm) * offset,
        y: (line.y[middle] ?? 0) - ((side * tx) / norm) * offset,
        heading,
        length,
        facing: side,
      };
      if (standIsClear(line, stand)) {
        stands.push(stand);
        break;
      }
    }
  }
  return stands;
}

function layoutTrees(line: Centerline, stands: readonly Stand[], seed: number): Tree[] {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < line.x.length; i++) {
    minX = Math.min(minX, line.x[i] ?? 0);
    maxX = Math.max(maxX, line.x[i] ?? 0);
    minY = Math.min(minY, line.y[i] ?? 0);
    maxY = Math.max(maxY, line.y[i] ?? 0);
  }
  const trees: Tree[] = [];
  const columns = Math.ceil((maxX - minX + TREE_AREA_MARGIN_M * 2) / TREE_GRID_M);
  const rows = Math.ceil((maxY - minY + TREE_AREA_MARGIN_M * 2) / TREE_GRID_M);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const x = minX - TREE_AREA_MARGIN_M + (column + hash(column, row, seed + 1)) * TREE_GRID_M;
      const y = minY - TREE_AREA_MARGIN_M + (row + hash(column, row, seed + 2)) * TREE_GRID_M;
      const density = groveNoise(x, y, seed);
      // Dense groves where the noise is high, a few lone trees elsewhere.
      if (hash(column, row, seed + 3) > (density > 0.58 ? 0.7 : 0.035)) continue;
      const scale = 0.7 + hash(column, row, seed + 4) * 0.75;
      const reach = TREE_RADIUS_M * scale;
      if (distancePastEdge(line, x, y, OFFTRACK_MARGIN_M) < DECOR_CLEARANCE_M + reach) continue;
      const nearStand = stands.some(
        (stand) => Math.hypot(stand.x - x, stand.y - y) < stand.length / 2 + 1.5 + reach,
      );
      if (nearStand) continue;
      trees.push({
        x,
        y,
        scale,
        rotation: hash(column, row, seed + 5) * Math.PI * 2,
        tint: hash(column, row, seed + 6),
      });
    }
  }
  // Keep the trees nearest to the circuit when there are too many: they are the ones in view.
  if (trees.length > MAX_TREES) {
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    trees.sort(
      (a, b) => Math.hypot(a.x - centerX, a.y - centerY) - Math.hypot(b.x - centerX, b.y - centerY),
    );
    trees.length = MAX_TREES;
  }
  return trees;
}

function layoutGantry(line: Centerline): Gantry {
  const start = startPose(line);
  const toRight = (line.right[0] ?? 0) + OFFTRACK_MARGIN_M + GANTRY_CLEARANCE_M;
  const toLeft = (line.left[0] ?? 0) + OFFTRACK_MARGIN_M + GANTRY_CLEARANCE_M;
  // Right of the direction of travel is (ty, -tx).
  return {
    right: { x: start.x + start.ty * toRight, y: start.y - start.tx * toRight },
    left: { x: start.x - start.ty * toLeft, y: start.y + start.tx * toLeft },
  };
}

/** Lays out the scenery of a track. `name` seeds the random choices: one track, one scenery. */
export function layoutDecor(line: Centerline, name: string): DecorLayout {
  const seed = seedOf(name);
  const stands = layoutStands(line);
  return { trees: layoutTrees(line, stands, seed), stands, gantry: layoutGantry(line) };
}
