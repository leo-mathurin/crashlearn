/**
 * Wall geometry from the occupancy image of a map. The simulator treats a pixel as occupied
 * when its value is 128 or less (`laser_models.py`): the LiDAR stops on those pixels. The walls
 * are the outline of that same set, extruded upwards, so that a LiDAR ray ends on a visible wall.
 *
 * Output coordinates are simulator metres (x east, y north) and a height z.
 */
import type { MapMeta } from "./track";

/** Values at or below this threshold are occupied, as in the simulator. */
const OCCUPIED_MAX = 128;
const ISO = OCCUPIED_MAX + 0.5;

export interface WallMesh {
  /** x, y, z triplets. */
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  /** Number of occupied pixels in the image. */
  occupiedPixels: number;
}

class MeshBuilder {
  positions: number[] = [];
  normals: number[] = [];
  indices: number[] = [];

  vertex(x: number, y: number, z: number, nx: number, ny: number, nz: number): number {
    this.positions.push(x, y, z);
    this.normals.push(nx, ny, nz);
    return this.positions.length / 3 - 1;
  }
}

/**
 * Extrudes the occupied pixels of a grayscale image (row 0 at the top) to `height` metres.
 * The outline follows the gray levels of the anti-aliased image (marching squares at the
 * occupancy threshold), so a diagonal wall is straight and not a staircase.
 */
export function buildWalls(
  gray: Uint8Array | Uint8ClampedArray,
  width: number,
  rows: number,
  meta: MapMeta,
  height: number,
): WallMesh {
  const builder = new MeshBuilder();
  const { resolution, originX, originY } = meta;
  // Value outside the image: free space.
  const value = (col: number, row: number): number =>
    col < 0 || row < 0 || col >= width || row >= rows ? 255 : (gray[row * width + col] ?? 255);
  // World position of a pixel centre. The simulator flips the image: row 0 is the north side.
  const worldX = (col: number): number => originX + (col + 0.5) * resolution;
  const worldY = (row: number): number => originY + (rows - 1 - row + 0.5) * resolution;

  let occupiedPixels = 0;
  for (let i = 0; i < gray.length; i++) if ((gray[i] ?? 255) <= OCCUPIED_MAX) occupiedPixels++;

  // Corner offsets of a cell, clockwise from the top-left pixel centre.
  const cornerCol = [0, 1, 1, 0];
  const cornerRow = [0, 0, 1, 1];
  const polygon: number[] = [];
  const crossing: boolean[] = [];

  for (let row = -1; row < rows; row++) {
    for (let col = -1; col < width; col++) {
      const v0 = value(col, row);
      const v1 = value(col + 1, row);
      const v2 = value(col + 1, row + 1);
      const v3 = value(col, row + 1);
      if (v0 >= ISO && v1 >= ISO && v2 >= ISO && v3 >= ISO) continue;
      const values = [v0, v1, v2, v3];
      const inside = values.map((v) => v < ISO);

      polygon.length = 0;
      crossing.length = 0;
      for (let corner = 0; corner < 4; corner++) {
        const next = (corner + 1) % 4;
        const c0 = col + (cornerCol[corner] ?? 0);
        const r0 = row + (cornerRow[corner] ?? 0);
        if (inside[corner]) {
          polygon.push(worldX(c0), worldY(r0));
          crossing.push(false);
        }
        if (inside[corner] !== inside[next]) {
          const from = values[corner] ?? 255;
          const to = values[next] ?? 255;
          const t = (ISO - from) / (to - from);
          const c1 = col + (cornerCol[next] ?? 0);
          const r1 = row + (cornerRow[next] ?? 0);
          polygon.push(
            worldX(c0) + (worldX(c1) - worldX(c0)) * t,
            worldY(r0) + (worldY(r1) - worldY(r0)) * t,
          );
          crossing.push(true);
        }
      }
      const count = polygon.length / 2;
      if (count < 3) continue;

      // Top face. The corners run clockwise seen from above (north up), so the fan is reversed
      // to face upwards.
      const first = builder.positions.length / 3;
      for (let i = 0; i < count; i++) {
        builder.vertex(polygon[i * 2] ?? 0, polygon[i * 2 + 1] ?? 0, height, 0, 0, 1);
      }
      for (let i = 1; i < count - 1; i++) builder.indices.push(first, first + i + 1, first + i);

      // Side faces: one quad for each edge of the polygon that is on the outline.
      for (let i = 0; i < count; i++) {
        const j = (i + 1) % count;
        if (!crossing[i] || !crossing[j]) continue;
        const ax = polygon[i * 2] ?? 0;
        const ay = polygon[i * 2 + 1] ?? 0;
        const bx = polygon[j * 2] ?? 0;
        const by = polygon[j * 2 + 1] ?? 0;
        const length = Math.hypot(bx - ax, by - ay);
        if (length < 1e-9) continue;
        // The polygon is clockwise: its outside is on the left of each edge.
        const nx = -(by - ay) / length;
        const ny = (bx - ax) / length;
        const base = builder.vertex(ax, ay, 0, nx, ny, 0);
        builder.vertex(bx, by, 0, nx, ny, 0);
        builder.vertex(bx, by, height, nx, ny, 0);
        builder.vertex(ax, ay, height, nx, ny, 0);
        builder.indices.push(base, base + 2, base + 1, base, base + 3, base + 2);
      }
    }
  }
  return {
    positions: Float32Array.from(builder.positions),
    normals: Float32Array.from(builder.normals),
    indices: Uint32Array.from(builder.indices),
    occupiedPixels,
  };
}
