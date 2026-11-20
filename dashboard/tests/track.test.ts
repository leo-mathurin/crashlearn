import { describe, expect, test } from "bun:test";
import { join } from "node:path";

import {
  type Centerline,
  corridor,
  distancePastEdge,
  edgeLine,
  lateralOffset,
  OFFTRACK_MARGIN_M,
  parseCenterline,
  parseMapYaml,
} from "../src/track/track";
import { buildWalls } from "../src/track/walls";

const MAPS = join(import.meta.dir, "../../vendor/simulation/maps");

async function centerlineOf(name: string): Promise<Centerline> {
  return parseCenterline(await Bun.file(join(MAPS, name, `${name}_centerline.csv`)).text());
}

describe("map files", () => {
  test("reads the resolution and the origin of a map", async () => {
    const meta = parseMapYaml(await Bun.file(join(MAPS, "Austin/Austin_map.yaml")).text());
    expect(meta.resolution).toBeCloseTo(0.08089);
    expect(meta.originX).toBeCloseTo(-21.2577, 3);
    expect(meta.originY).toBeCloseTo(-70.804, 3);
  });

  test("refuses a map file without resolution", () => {
    expect(() => parseMapYaml("image: x.png\n")).toThrow();
  });

  test("closes the centerline like the simulator and keeps the measured widths", async () => {
    const line = await centerlineOf("Austin");
    expect(line.closed).toBe(true);
    expect(line.x.length).toBe(1103);
    expect(line.x.at(-1)).toBe(line.x[0]);
    expect(line.right[0]).toBeCloseTo(1.1021);
    expect(line.left[0]).toBeCloseTo(0.9808);
    // Length of the lap in docs/tracks.md.
    expect(line.arc.at(-1)).toBeCloseTo(421.04, 1);
  });
});

describe("track edges", () => {
  // A straight line to the east: the right side is the south.
  const straight = parseCenterline("0,0,1,2\n10,0,1,2\n20,0,1,2\n30,0,1,2\n");

  test("measures the lateral offset with the sign of the simulator: positive on the right", () => {
    expect(lateralOffset(straight, 5, -0.5).lateral).toBeCloseTo(0.5);
    expect(lateralOffset(straight, 5, 0.5).lateral).toBeCloseTo(-0.5);
    expect(lateralOffset(straight, 25, 0).segment).toBe(2);
  });

  test("applies the off-track rule: edge plus 1 m, on each side with its own width", () => {
    expect(distancePastEdge(straight, 5, -1.9, OFFTRACK_MARGIN_M)).toBeLessThan(0);
    expect(distancePastEdge(straight, 5, -2.1, OFFTRACK_MARGIN_M)).toBeGreaterThan(0);
    expect(distancePastEdge(straight, 5, 2.9, OFFTRACK_MARGIN_M)).toBeLessThan(0);
    expect(distancePastEdge(straight, 5, 3.1, OFFTRACK_MARGIN_M)).toBeGreaterThan(0);
  });

  test("puts the left edge on the left and the right edge on the right", () => {
    const { left, right } = corridor(straight);
    expect(left[1]).toBeCloseTo(2);
    expect(right[1]).toBeCloseTo(-1);
  });

  test.each(["Austin", "Spielberg", "Montreal"])(
    "draws the DNF limit of %s where the simulator ends the race",
    async (name) => {
      const line = await centerlineOf(name);
      for (const side of ["left", "right"] as const) {
        const limit = edgeLine(line, side, OFFTRACK_MARGIN_M);
        const count = limit.points.length / 2;
        // Most waypoints give a point; only the folds inside tight turns are removed.
        expect(count).toBeGreaterThan((line.x.length - 1) * 0.85);
        for (let i = 0; i < count; i++) {
          const x = limit.points[i * 2] ?? 0;
          const y = limit.points[i * 2 + 1] ?? 0;
          const distance = distancePastEdge(line, x, y, OFFTRACK_MARGIN_M);
          // On the limit: never well inside the allowed area, never far outside.
          expect(distance).toBeGreaterThan(-0.061);
          expect(distance).toBeLessThan(0.15);
        }
      }
    },
  );
});

describe("walls", () => {
  const size = 24;
  const meta = { resolution: 0.1, originX: -1, originY: -2 };
  const height = 0.25;
  // A diagonal wall two pixels thick.
  const gray = new Uint8Array(size * size).fill(255);
  for (let i = 4; i < 20; i++) {
    gray[i * size + i] = 0;
    gray[i * size + i + 1] = 0;
  }
  const walls = buildWalls(gray, size, size, meta, height);

  test("counts the occupied pixels with the threshold of the simulator", () => {
    const levels = Uint8Array.from([0, 128, 129, 255]);
    expect(buildWalls(levels, 2, 2, meta, height).occupiedPixels).toBe(2);
    expect(walls.occupiedPixels).toBe(32);
  });

  test("builds faces whose winding agrees with their normal", () => {
    const { positions, normals, indices } = walls;
    expect(indices.length).toBeGreaterThan(0);
    for (let t = 0; t < indices.length; t += 3) {
      const [a, b, c] = [indices[t] ?? 0, indices[t + 1] ?? 0, indices[t + 2] ?? 0];
      const p = (v: number, k: number): number => positions[v * 3 + k] ?? 0;
      const ux = p(b, 0) - p(a, 0);
      const uy = p(b, 1) - p(a, 1);
      const uz = p(b, 2) - p(a, 2);
      const vx = p(c, 0) - p(a, 0);
      const vy = p(c, 1) - p(a, 1);
      const vz = p(c, 2) - p(a, 2);
      const cross = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
      const dot =
        (cross[0] ?? 0) * (normals[a * 3] ?? 0) +
        (cross[1] ?? 0) * (normals[a * 3 + 1] ?? 0) +
        (cross[2] ?? 0) * (normals[a * 3 + 2] ?? 0);
      expect(dot).toBeGreaterThan(0);
    }
  });

  test("stays within one pixel of the occupied pixels and at the wall height", () => {
    const { positions } = walls;
    for (let v = 0; v < positions.length / 3; v++) {
      const x = positions[v * 3] ?? 0;
      const y = positions[v * 3 + 1] ?? 0;
      const z = positions[v * 3 + 2] ?? 0;
      expect(z === 0 || Math.abs(z - height) < 1e-6).toBe(true);
      // Back to pixel coordinates: the nearest occupied pixel centre is at most one pixel away.
      const col = (x - meta.originX) / meta.resolution - 0.5;
      const row = size - 1 - ((y - meta.originY) / meta.resolution - 0.5);
      let nearest = Infinity;
      for (let i = 4; i < 20; i++) {
        nearest = Math.min(nearest, Math.hypot(col - i, row - i), Math.hypot(col - i - 1, row - i));
      }
      expect(nearest).toBeLessThanOrEqual(1);
    }
  });

  test("puts the north of the image at the top: row 0 has the largest y", () => {
    const top = new Uint8Array(size * size).fill(255);
    top[3] = 0;
    const mesh = buildWalls(top, size, size, meta, height);
    const ys = Array.from(
      { length: mesh.positions.length / 3 },
      (_, v) => mesh.positions[v * 3 + 1] ?? 0,
    );
    const expected = meta.originY + (size - 0.5) * meta.resolution;
    expect(Math.max(...ys)).toBeLessThan(expected + meta.resolution);
    expect(Math.min(...ys)).toBeGreaterThan(expected - meta.resolution);
  });
});
