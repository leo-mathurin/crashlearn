import { expect, type Page, test } from "@playwright/test";

import { failOnErrors, openReplay, press, state } from "./dashboard";

test.beforeEach(({ page }) => failOnErrors(page));

/** Waits for the end of the move from one camera to another. */
async function cameraSettled(page: Page): Promise<void> {
  await expect.poll(async () => (await state(page)).cameraMoving).toBe(false);
}

function distance(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

test("changes the camera without moving the playhead", async ({ page }) => {
  await openReplay(page, "demo-4-voitures-spielberg");
  await press(page, "Shift+ArrowRight", "Shift+ArrowRight");
  const before = await state(page);
  expect(before.camera).toBe("Aérienne");

  for (const name of ["Libre", "Poursuite", "Aérienne"]) {
    await page.getByRole("radio", { name }).click();
    await expect(page.getByRole("radio", { name })).toHaveAttribute("aria-checked", "true");
    // A click starts a glide from the old camera to the new one.
    expect((await state(page)).cameraMoving).toBe(true);
    await cameraSettled(page);
    const now = await state(page);
    expect(now.camera).toBe(name);
    expect(now.time).toBe(before.time);
    expect(now.playing).toBe(false);
  }
  // Back on the Aerial camera, the view is the one that was left.
  const after = await state(page);
  expect(after.view.x).toBeCloseTo(before.view.x, 3);
  expect(after.view.pixelsPerMetre).toBeCloseTo(before.view.pixelsPerMetre, 3);
});

test("changes the camera at once from the keyboard", async ({ page }) => {
  await openReplay(page, "demo-solo-spielberg");
  for (const [key, name] of [
    ["f", "Libre"],
    ["c", "Poursuite"],
    ["a", "Aérienne"],
  ] as const) {
    await press(page, key);
    const now = await state(page);
    expect(now.camera).toBe(name);
    expect(now.cameraMoving).toBe(false);
  }
});

test("orbits, zooms and moves the orbit point of the Free camera", async ({ page }) => {
  await openReplay(page, "demo-solo-spielberg");
  await press(page, "f");
  const first = await state(page);
  // A perspective camera above the ground, south of the point it looks at.
  expect(first.cameraPosition.z).toBeGreaterThan(5);
  expect(first.cameraPosition.y).toBeLessThan(first.view.y);

  await page.mouse.move(640, 400);
  await page.mouse.down();
  await page.mouse.move(790, 400, { steps: 6 });
  await page.mouse.up();
  await expect
    .poll(async () => Math.abs((await state(page)).cameraPosition.x - first.cameraPosition.x))
    .toBeGreaterThan(5);
  const orbited = await state(page);
  // An orbit turns around the same point.
  expect(distance(orbited.view, first.view)).toBeLessThan(0.5);

  await page.mouse.wheel(0, -900);
  await expect
    .poll(async () => (await state(page)).view.pixelsPerMetre)
    .toBeGreaterThan(orbited.view.pixelsPerMetre * 1.3);

  await page.mouse.dblclick(900, 560);
  await expect
    .poll(async () => distance((await state(page)).view, orbited.view))
    .toBeGreaterThan(3);
});

test("follows the selected car with the Chase camera", async ({ page }) => {
  await openReplay(page, "demo-4-voitures-spielberg");
  await press(page, ...Array.from({ length: 5 }, () => "Shift+ArrowRight"), "c");
  const now = await state(page);
  expect(now.followedCar).toBe(0);
  expect(now.followLinked).toBe(true);
  const car = now.carPositions[0];
  if (!car) throw new Error("car 0 is not active");
  // Behind the car, low above the ground, and looking at it.
  expect(distance(now.cameraPosition, car)).toBeLessThan(4);
  expect(now.cameraPosition.z).toBeLessThan(2);
  expect(distance(now.view, car)).toBeLessThan(2);

  // Linked: the camera goes to the car that the leaderboard selects.
  await page.getByRole("listitem").filter({ hasText: "2-driver2" }).click();
  await expect.poll(async () => (await state(page)).followedCar).toBe(1);
  await expect
    .poll(async () => {
      const next = await state(page);
      const target = next.carPositions[1];
      return target ? distance(next.cameraPosition, target) : Infinity;
    })
    .toBeLessThan(4);
});

test("unlinks and links again the followed car with the Suivre menu", async ({ page }) => {
  await openReplay(page, "demo-4-voitures-spielberg");
  await press(page, "c");
  const follow = page.getByRole("button", { name: /^Suivre/ });
  await expect(follow).toHaveText("Suivre : 1-driver1");

  await follow.click();
  await page.getByRole("menuitemradio", { name: "3-driver1b" }).click();
  await expect(follow).toHaveText("Suivre : 3-driver1b (déliée)");
  let now = await state(page);
  expect(now.followedCar).toBe(2);
  expect(now.followLinked).toBe(false);
  expect(now.selectedCar).toBe(0);

  // Unlinked: a new selected car does not move the camera.
  await press(page, "2");
  now = await state(page);
  expect(now.selectedCar).toBe(1);
  expect(now.followedCar).toBe(2);

  await follow.click();
  await page.getByRole("menuitemradio", { name: "Voiture sélectionnée" }).click();
  await expect(follow).toHaveText("Suivre : 2-driver2");
  now = await state(page);
  expect(now.followLinked).toBe(true);
  expect(now.followedCar).toBe(1);
});

test("stays 2 s on a followed car that becomes inactive, then follows the best active car", async ({
  page,
}) => {
  await openReplay(page, "demo-dnf-sochi");
  // Car 4 has a DNF at frame 445. Go to frame 440 and follow it.
  await press(page, ...Array.from({ length: 22 }, () => "Shift+ArrowRight"), "4", "c");
  const before = await state(page);
  expect(before.frame).toBe(440);
  expect(before.followedCar).toBe(3);
  const lastSeen = before.carPositions[3];
  if (!lastSeen) throw new Error("car 4 should still be active");

  await page.getByRole("button", { name: "Lecture" }).click();
  await expect.poll(async () => (await state(page)).dnfMarkers).toContain(3);
  const inactiveAt = Date.now();
  // During the 2 s, the camera is still on the place of the DNF.
  await page.waitForTimeout(1000);
  const held = await state(page);
  expect(held.followedCar).toBe(3);
  expect(distance(held.cameraPosition, lastSeen)).toBeLessThan(5);

  await expect.poll(async () => (await state(page)).followedCar, { timeout: 6000 }).not.toBe(3);
  expect(Date.now() - inactiveAt).toBeGreaterThan(1700);
  const after = await state(page);
  // The best-ranked car that still races, and the selection follows because it is linked.
  const bestActive = after.order.find((car) => after.carPositions[car] !== null);
  expect(after.followedCar).toBe(bestActive);
  expect(after.selectedCar).toBe(bestActive);
  expect(after.followLinked).toBe(true);
});

test("keeps the camera on an inactive car that the user chooses", async ({ page }) => {
  await openReplay(page, "demo-dnf-sochi");
  await press(page, "End", "c", "4");
  await page.waitForTimeout(2600);
  const now = await state(page);
  expect(now.followedCar).toBe(3);
  expect(now.dnfMarkers).toContain(3);
});

test("shows the mini-map with every camera", async ({ page }) => {
  await openReplay(page, "demo-dnf-sochi");
  const map = page.getByRole("img", { name: "Mini-carte du circuit" });
  /** Number of different colours on the mini-map: a picture of the circuit has many. */
  const colours = (): Promise<number> =>
    map.locator("canvas").evaluate((canvas: HTMLCanvasElement) => {
      const context = canvas.getContext("2d");
      if (!context) return 0;
      const { data } = context.getImageData(0, 0, canvas.width, canvas.height);
      const seen = new Set<number>();
      for (let i = 0; i < data.length; i += 4 * 7) {
        seen.add(((data[i] ?? 0) << 16) | ((data[i + 1] ?? 0) << 8) | (data[i + 2] ?? 0));
      }
      return seen.size;
    });
  for (const key of ["a", "f", "c"]) {
    await press(page, key);
    await expect(map).toBeVisible();
    expect(await colours()).toBeGreaterThan(50);
  }
});

test("moves the Free camera over the ground with the move keys", async ({ page }) => {
  await openReplay(page, "demo-solo-spielberg");
  await press(page, "f");
  const start = await state(page);

  // The keys are read by their place: KeyW is Z on an AZERTY keyboard.
  await page.keyboard.down("KeyW");
  await page.waitForTimeout(400);
  await page.keyboard.up("KeyW");
  await press(page);
  const ahead = await state(page);
  // The first view looks north: forward is +y, and the camera moves with its orbit centre.
  expect(ahead.view.y - start.view.y).toBeGreaterThan(1);
  expect(ahead.view.x).toBeCloseTo(start.view.x, 1);
  expect(ahead.cameraPosition.y - start.cameraPosition.y).toBeGreaterThan(1);
  expect(ahead.cameraPosition.z).toBeCloseTo(start.cameraPosition.z, 3);

  await page.keyboard.down("KeyD");
  await page.waitForTimeout(400);
  await page.keyboard.up("KeyD");
  await press(page);
  const right = await state(page);
  expect(right.view.x - ahead.view.x).toBeGreaterThan(1);
  // The move key is not the shortcut of the scenery in this camera.
  expect(right.decorVisible).toBe(ahead.decorVisible);
  expect(right.camera).toBe("Libre");

  // The camera stops after the key is up.
  await expect
    .poll(async () => {
      const before = await state(page);
      await page.waitForTimeout(250);
      return Math.abs((await state(page)).view.x - before.view.x);
    })
    .toBeLessThan(0.005);
});
