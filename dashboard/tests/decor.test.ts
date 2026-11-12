import { describe, expect, test } from "bun:test";
import { join } from "node:path";

import {
  DECOR_CLEARANCE_M,
  GANTRY_CLEARANCE_M,
  layoutDecor,
  standCorners,
} from "../src/track/decorLayout";
import { distancePastEdge, OFFTRACK_MARGIN_M, parseCenterline } from "../src/track/track";

const MAPS = join(import.meta.dir, "../../vendor/simulation/maps");

describe.each(["Austin", "Spielberg", "Sochi", "Montreal", "Monza"])("scenery of %s", (name) => {
  const linePromise = Bun.file(join(MAPS, name, `${name}_centerline.csv`))
    .text()
    .then(parseCenterline);

  test("keeps every tree outside the DNF limit, with its clearance", async () => {
    const line = await linePromise;
    const { trees } = layoutDecor(line, name);
    expect(trees.length).toBeGreaterThan(50);
    expect(trees.length).toBeLessThanOrEqual(900);
    for (const tree of trees) {
      // Positive: outside the limit of the simulator. The canopy of the tree stays outside too.
      const outside = distancePastEdge(line, tree.x, tree.y, OFFTRACK_MARGIN_M);
      expect(outside).toBeGreaterThanOrEqual(DECOR_CLEARANCE_M);
    }
  });

  test("keeps every grandstand outside the DNF limit, with its clearance", async () => {
    const line = await linePromise;
    const { stands } = layoutDecor(line, name);
    expect(stands.length).toBeLessThanOrEqual(3);
    for (const stand of stands) {
      const corners = standCorners(stand);
      for (let i = 0; i < 4; i++) {
        const outside = distancePastEdge(
          line,
          corners[i * 2] ?? 0,
          corners[i * 2 + 1] ?? 0,
          OFFTRACK_MARGIN_M,
        );
        expect(outside).toBeGreaterThanOrEqual(DECOR_CLEARANCE_M);
      }
    }
  });

  test("puts the posts of the start gantry outside the DNF limit, on each side", async () => {
    const line = await linePromise;
    const { gantry } = layoutDecor(line, name);
    if (!gantry) throw new Error("no gantry");
    for (const post of [gantry.left, gantry.right]) {
      const outside = distancePastEdge(line, post.x, post.y, OFFTRACK_MARGIN_M);
      expect(outside).toBeGreaterThan(GANTRY_CLEARANCE_M - 0.1);
    }
  });

  test("gives the same scenery at each load", async () => {
    const line = await linePromise;
    expect(layoutDecor(line, name)).toEqual(layoutDecor(line, name));
  });
});
