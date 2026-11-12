/**
 * Meshes of a track, in simulator coordinates (x east, y north, z up, metres). Everything here
 * comes from the map data: asphalt and painted edges from the centerline and its widths, walls
 * from the occupancy image, DNF limit from the off-track rule of the simulator.
 */
import * as THREE from "three/webgpu";

import type { TrackAssets } from "../track/assets";
import {
  type Bounds,
  boundsOf,
  type Centerline,
  corridor,
  curvature,
  type EdgeLine,
  edgeLine,
  OFFTRACK_MARGIN_M,
  startPose,
} from "../track/track";
import { buildWalls } from "../track/walls";
import { asphalt, concrete, grass } from "./textures";

export const WALL_HEIGHT_M = 0.25;

/** Heights of the flat layers above the ground, so that they never fight for depth. */
const ASPHALT_Z = 0.006;
const PAINT_Z = 0.011;
const LIMIT_Z = 0.016;

const ASPHALT_TILE_M = 2.4;
const GRASS_TILE_M = 11;
const CONCRETE_TILE_M = 1.2;

/** The walls stand on the track edge: the paint starts this far inside, clear of them. */
const PAINT_INSET_M = 0.1;
const EDGE_LINE_WIDTH_M = 0.07;
const KERB_WIDTH_M = 0.22;
const KERB_STRIPE_M = 0.45;
/** A painted edge becomes a red and white kerb where the centerline turns tighter than this. */
const KERB_CURVATURE_RADPM = 0.14;

export interface TrackView {
  group: THREE.Group;
  /** Dashed line of the off-track DNF limit: the track edge plus 1 m. */
  limit: THREE.Object3D;
  /** Bounds of the DNF limit, in simulator metres: the area a car can be in. */
  bounds: Bounds;
  triangles: number;
  dispose(): void;
}

function flatMaterial(parameters: THREE.MeshStandardMaterialParameters, offset: number) {
  return new THREE.MeshStandardMaterial({
    ...parameters,
    polygonOffset: true,
    polygonOffsetFactor: -offset,
    polygonOffsetUnits: -offset,
  });
}

function buildGround(bounds: Bounds, anisotropy: number): THREE.Mesh {
  const margin = 400;
  const width = bounds.maxX - bounds.minX + margin * 2;
  const depth = bounds.maxY - bounds.minY + margin * 2;
  const surface = grass(anisotropy);
  for (const texture of [surface.map, surface.normalMap]) {
    texture.repeat.set(width / GRASS_TILE_M, depth / GRASS_TILE_M);
  }
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(width, depth),
    new THREE.MeshStandardMaterial({ ...surface, roughness: 0.95, metalness: 0 }),
  );
  mesh.position.set((bounds.minX + bounds.maxX) / 2, (bounds.minY + bounds.maxY) / 2, 0);
  mesh.receiveShadow = true;
  return mesh;
}

function buildAsphalt(line: Centerline, anisotropy: number): THREE.Mesh {
  const { left, right } = corridor(line);
  const count = line.x.length;
  const positions = new Float32Array(count * 6);
  const uvs = new Float32Array(count * 4);
  const indices: number[] = [];
  for (let i = 0; i < count; i++) {
    positions.set([left[i * 2] ?? 0, left[i * 2 + 1] ?? 0, ASPHALT_Z], i * 6);
    positions.set([right[i * 2] ?? 0, right[i * 2 + 1] ?? 0, ASPHALT_Z], i * 6 + 3);
    // World-space UVs: the grain has the same size everywhere, with no stretch in the turns.
    uvs.set(
      [
        (left[i * 2] ?? 0) / ASPHALT_TILE_M,
        (left[i * 2 + 1] ?? 0) / ASPHALT_TILE_M,
        (right[i * 2] ?? 0) / ASPHALT_TILE_M,
        (right[i * 2 + 1] ?? 0) / ASPHALT_TILE_M,
      ],
      i * 4,
    );
    if (i < count - 1) {
      const a = i * 2;
      indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  const normals = geometry.getAttribute("normal");
  // Inside a tight turn the ribbon folds over itself: keep every normal upwards.
  for (let i = 0; i < normals.count; i++) normals.setXYZ(i, 0, 0, 1);
  const mesh = new THREE.Mesh(
    geometry,
    flatMaterial(
      { ...asphalt(anisotropy), roughness: 0.82, metalness: 0, side: THREE.DoubleSide },
      1,
    ),
  );
  mesh.receiveShadow = true;
  return mesh;
}

/** Appends a flat quad a-b-c-d (two triangles facing up) with one colour. */
function pushQuad(
  target: { positions: number[]; colors: number[] },
  corners: readonly number[],
  z: number,
  color: THREE.Color,
): void {
  const order = [0, 1, 2, 0, 2, 3];
  for (const corner of order) {
    target.positions.push(corners[corner * 2] ?? 0, corners[corner * 2 + 1] ?? 0, z);
    target.colors.push(color.r, color.g, color.b);
  }
}

function coloredMesh(
  data: { positions: number[]; colors: number[] },
  material: THREE.Material,
): THREE.Mesh {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(data.positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(data.colors, 3));
  const normals = new Float32Array(data.positions.length);
  for (let i = 2; i < normals.length; i += 3) normals[i] = 1;
  geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  return new THREE.Mesh(geometry, material);
}

/**
 * Walks an edge line and calls `emit` for each segment with the two ends, the unit normal that
 * points to the left of the direction of travel, and the arc length at the start.
 */
function walkEdge(
  edge: EdgeLine,
  emit: (
    ax: number,
    ay: number,
    bx: number,
    by: number,
    nx: number,
    ny: number,
    arc: number,
    index: number,
  ) => void,
): void {
  const count = edge.points.length / 2;
  const segments = edge.closed ? count : count - 1;
  let arc = 0;
  for (let i = 0; i < segments; i++) {
    const j = (i + 1) % count;
    const ax = edge.points[i * 2] ?? 0;
    const ay = edge.points[i * 2 + 1] ?? 0;
    const bx = edge.points[j * 2] ?? 0;
    const by = edge.points[j * 2 + 1] ?? 0;
    const length = Math.hypot(bx - ax, by - ay);
    if (length < 1e-6) continue;
    emit(ax, ay, bx, by, -(by - ay) / length, (bx - ax) / length, arc, i);
    arc += length;
  }
}

/**
 * Cuts the arc range of a segment at the marks of a repeating pattern. `marks` are ascending
 * arc lengths in one period, the last one is the period. `emit` gets each piece as fractions
 * of the segment, with the index of the mark interval it is in.
 */
function cutByPattern(
  arc: number,
  length: number,
  marks: readonly number[],
  emit: (from: number, to: number, interval: number) => void,
): void {
  const period = marks.at(-1) ?? 1;
  const end = arc + length;
  for (let base = Math.floor(arc / period) * period; base < end; base += period) {
    let previous = 0;
    marks.forEach((mark, interval) => {
      const from = Math.max(arc, base + previous);
      const to = Math.min(end, base + mark);
      previous = mark;
      if (to - from > 1e-6) emit((from - arc) / length, (to - arc) / length, interval);
    });
  }
}

/** Painted track edges: a white line, and red and white kerb stripes in the tight turns. */
function buildPaint(line: Centerline): THREE.Mesh {
  const data = { positions: [] as number[], colors: [] as number[] };
  const white = new THREE.Color(0xe9e9e4);
  const red = new THREE.Color(0xc22a25);
  const bend = curvature(line);
  for (const side of ["left", "right"] as const) {
    const edge = edgeLine(line, side, -PAINT_INSET_M);
    // The paint is inside the track: to the right of travel on the left edge, and the reverse.
    const inward = side === "left" ? -1 : 1;
    walkEdge(edge, (ax, ay, bx, by, nx, ny, arc, index) => {
      const kerb = (bend[edge.waypoint[index] ?? 0] ?? 0) > KERB_CURVATURE_RADPM;
      const width = (kerb ? KERB_WIDTH_M : EDGE_LINE_WIDTH_M) * inward;
      const length = Math.hypot(bx - ax, by - ay);
      cutByPattern(arc, length, [KERB_STRIPE_M, KERB_STRIPE_M * 2], (from, to, interval) => {
        const x0 = ax + (bx - ax) * from;
        const y0 = ay + (by - ay) * from;
        const x1 = ax + (bx - ax) * to;
        const y1 = ay + (by - ay) * to;
        pushQuad(
          data,
          [x0, y0, x1, y1, x1 + nx * width, y1 + ny * width, x0 + nx * width, y0 + ny * width],
          PAINT_Z,
          kerb && interval === 0 ? red : white,
        );
      });
    });
  }
  const mesh = coloredMesh(
    data,
    flatMaterial({ vertexColors: true, roughness: 0.55, side: THREE.DoubleSide }, 2),
  );
  mesh.receiveShadow = true;
  return mesh;
}

/** Chequered band across the track at the first waypoint: the start pose of the simulator. */
function buildStartLine(line: Centerline): THREE.Mesh {
  const data = { positions: [] as number[], colors: [] as number[] };
  const start = startPose(line);
  const square = 0.15;
  const light = new THREE.Color(0xf2f2ee);
  const dark = new THREE.Color(0x121214);
  const toRight = line.right[0] ?? 0;
  const toLeft = line.left[0] ?? 0;
  const columns = Math.round((toRight + toLeft) / square);
  const step = (toRight + toLeft) / columns;
  // Lateral axis: from the left edge to the right edge. Right of travel is (ty, -tx).
  const lx = start.ty;
  const ly = -start.tx;
  for (let column = 0; column < columns; column++) {
    for (let rowIndex = 0; rowIndex < 2; rowIndex++) {
      const u0 = -toLeft + column * step;
      const v0 = (rowIndex - 1) * square;
      const corner = (u: number, v: number): [number, number] => [
        start.x + lx * u + start.tx * v,
        start.y + ly * u + start.ty * v,
      ];
      pushQuad(
        data,
        [
          ...corner(u0, v0),
          ...corner(u0 + step, v0),
          ...corner(u0 + step, v0 + square),
          ...corner(u0, v0 + square),
        ],
        PAINT_Z,
        (column + rowIndex) % 2 === 0 ? light : dark,
      );
    }
  }
  const mesh = coloredMesh(
    data,
    flatMaterial({ vertexColors: true, roughness: 0.5, side: THREE.DoubleSide }, 3),
  );
  mesh.receiveShadow = true;
  return mesh;
}

function buildWallMesh(assets: TrackAssets, anisotropy: number): THREE.Mesh {
  const { image, meta } = assets;
  const walls = buildWalls(image.gray, image.width, image.height, meta, WALL_HEIGHT_M);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(walls.positions, 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(walls.normals, 3));
  const uvs = new Float32Array((walls.positions.length / 3) * 2);
  for (let i = 0; i < walls.positions.length / 3; i++) {
    const x = walls.positions[i * 3] ?? 0;
    const y = walls.positions[i * 3 + 1] ?? 0;
    const z = walls.positions[i * 3 + 2] ?? 0;
    const nx = walls.normals[i * 3] ?? 0;
    const ny = walls.normals[i * 3 + 1] ?? 0;
    const top = (walls.normals[i * 3 + 2] ?? 0) > 0.5;
    // Top faces are mapped from above, side faces along the wall.
    uvs[i * 2] = (top ? x : x * -ny + y * nx) / CONCRETE_TILE_M;
    uvs[i * 2 + 1] = (top ? y : z) / CONCRETE_TILE_M;
  }
  geometry.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  geometry.setIndex(new THREE.BufferAttribute(walls.indices, 1));
  const mesh = new THREE.Mesh(
    geometry,
    new THREE.MeshStandardMaterial({ ...concrete(anisotropy), roughness: 0.78, metalness: 0 }),
  );
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

/** Dashed lines at the off-track DNF limit, on both sides of the track. */
function buildLimit(line: Centerline): { mesh: THREE.Mesh; bounds: Bounds } {
  const data = { positions: [] as number[], colors: [] as number[] };
  const color = new THREE.Color(0xff5a47);
  const half = 0.035;
  const dash = 0.6;
  const period = 1.0;
  const all: number[] = [];
  for (const side of ["left", "right"] as const) {
    const edge = edgeLine(line, side, OFFTRACK_MARGIN_M);
    for (const value of edge.points) all.push(value);
    walkEdge(edge, (ax, ay, bx, by, nx, ny, arc) => {
      const length = Math.hypot(bx - ax, by - ay);
      cutByPattern(arc, length, [dash, period], (from, to, interval) => {
        if (interval !== 0) return;
        const x0 = ax + (bx - ax) * from;
        const y0 = ay + (by - ay) * from;
        const x1 = ax + (bx - ax) * to;
        const y1 = ay + (by - ay) * to;
        pushQuad(
          data,
          [
            x0 - nx * half,
            y0 - ny * half,
            x1 - nx * half,
            y1 - ny * half,
            x1 + nx * half,
            y1 + ny * half,
            x0 + nx * half,
            y0 + ny * half,
          ],
          LIMIT_Z,
          color,
        );
      });
    });
  }
  const mesh = coloredMesh(
    data,
    new THREE.MeshBasicMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -4,
      polygonOffsetUnits: -4,
    }),
  );
  return { mesh, bounds: boundsOf(all) };
}

function triangleCount(object: THREE.Object3D): number {
  let total = 0;
  object.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    const geometry: THREE.BufferGeometry = child.geometry;
    total += (geometry.index?.count ?? geometry.getAttribute("position").count) / 3;
  });
  return total;
}

export function buildTrackView(assets: TrackAssets, anisotropy: number): TrackView {
  const group = new THREE.Group();
  group.name = `track:${assets.name}`;
  const limit = buildLimit(assets.centerline);
  limit.mesh.name = "dnf-limit";
  group.add(
    buildGround(limit.bounds, anisotropy),
    buildAsphalt(assets.centerline, anisotropy),
    buildPaint(assets.centerline),
    buildStartLine(assets.centerline),
    buildWallMesh(assets, anisotropy),
    limit.mesh,
  );
  return {
    group,
    limit: limit.mesh,
    bounds: limit.bounds,
    triangles: triangleCount(group),
    dispose() {
      group.traverse((child) => {
        if (!(child instanceof THREE.Mesh)) return;
        child.geometry.dispose();
        const material: THREE.Material & { map?: THREE.Texture; normalMap?: THREE.Texture } =
          child.material;
        material.map?.dispose();
        material.normalMap?.dispose();
        material.dispose();
      });
    },
  };
}

/** Plain ground for a replay whose map is not available: a neutral grid, no invented circuit. */
export function buildFallbackGround(bounds: Bounds): TrackView {
  const group = new THREE.Group();
  group.name = "track:unavailable";
  const size = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY) + 40;
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(size * 4, size * 4),
    new THREE.MeshStandardMaterial({ color: 0x23262b, roughness: 1 }),
  );
  plane.position.set((bounds.minX + bounds.maxX) / 2, (bounds.minY + bounds.maxY) / 2, 0);
  plane.receiveShadow = true;
  const grid = new THREE.GridHelper(
    Math.ceil(size / 10) * 10,
    Math.ceil(size / 10),
    0x5b616b,
    0x3a3f47,
  );
  // GridHelper lies in the XZ plane: turn it to the XY plane of the simulator.
  grid.rotation.x = Math.PI / 2;
  grid.position.set(plane.position.x, plane.position.y, 0.005);
  group.add(plane, grid);
  return {
    group,
    limit: new THREE.Group(),
    bounds,
    triangles: 2,
    dispose() {
      plane.geometry.dispose();
      grid.geometry.dispose();
    },
  };
}
