import { expect, type Page, test } from "@playwright/test";

import { failOnErrors, openReplay, press, renderedShare, state } from "./dashboard";

test.beforeEach(({ page }) => failOnErrors(page));

/**
 * Share of the screen pixels that have the colour of the LiDAR beams of the selected car. The
 * beams are the only thing of that pale green in the scene.
 */
function lidarShare(page: Page): Promise<number> {
  return page.evaluate(
    () =>
      new Promise<number>((resolve) => {
        requestAnimationFrame(() => {
          const scene = document.querySelector<HTMLCanvasElement>("#scene");
          const copy = document.createElement("canvas");
          copy.width = 640;
          copy.height = 400;
          const context = copy.getContext("2d");
          if (!scene || !context) throw new Error("no canvas");
          context.drawImage(scene, 0, 0, copy.width, copy.height);
          const { data } = context.getImageData(0, 0, copy.width, copy.height);
          let beams = 0;
          for (let i = 0; i < data.length; i += 4) {
            const [red = 0, green = 0, blue = 0] = [data[i], data[i + 1], data[i + 2]];
            if (green > 150 && green - red > 45 && green - blue > 20) beams++;
          }
          resolve(beams / (data.length / 4));
        });
      }),
  );
}

test("draws the scenery by default and removes it in one click", async ({ page }) => {
  await openReplay(page, "demo-4-voitures-spielberg");
  const toggle = page.getByRole("switch", { name: "Décor" });
  await expect(toggle).toHaveAttribute("aria-checked", "true");
  const withDecor = await state(page);
  expect(withDecor.decorVisible).toBe(true);
  // Trees, grandstands, gantry, sky.
  expect(withDecor.decorTriangles).toBeGreaterThan(20_000);
  expect(withDecor.triangles).toBeGreaterThan(withDecor.decorTriangles);

  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-checked", "false");
  const without = await state(page);
  expect(without.decorVisible).toBe(false);
  expect(without.triangles).toBe(withDecor.triangles - withDecor.decorTriangles);
  expect(await renderedShare(page)).toBeGreaterThan(0.5);
});

test("remembers that the scenery is off", async ({ page }) => {
  await openReplay(page, "demo-solo-spielberg");
  await press(page, "g");
  expect((await state(page)).decorVisible).toBe(false);
  await openReplay(page, "demo-solo-spielberg");
  expect((await state(page)).decorVisible).toBe(false);
  await expect(page.getByRole("switch", { name: "Décor" })).toHaveAttribute(
    "aria-checked",
    "false",
  );
});

test("never hides the LiDAR rays behind the scenery, whatever the camera", async ({ page }) => {
  await openReplay(page, "demo-4-voitures-spielberg");
  await press(page, ...Array.from({ length: 8 }, () => "Shift+ArrowRight"));
  for (const key of ["a", "f", "c"]) {
    await press(page, key);
    // The scenery is drawn first and the race over it: with or without the scenery, the same
    // pixels show the beams.
    const withDecor = await lidarShare(page);
    await press(page, "g");
    const without = await lidarShare(page);
    await press(page, "g");
    expect(withDecor).toBeGreaterThan(0);
    expect(Math.abs(withDecor - without)).toBeLessThan(Math.max(without * 0.03, 0.0005));
  }
});

test("draws a sky for a replay whose map is missing", async ({ page }) => {
  await openReplay(page, "unknown-map");
  const ready = await state(page);
  expect(ready.mapAvailable).toBe(false);
  expect(ready.decorVisible).toBe(true);
  // No circuit to lay trees around: the sky, the clouds and the hills only.
  expect(ready.decorTriangles).toBeGreaterThan(0);
  expect(ready.decorTriangles).toBeLessThan(25_000);
});
