import { expect, test } from "@playwright/test";

import { failOnErrors, openReplay, press, renderedShare, state } from "./dashboard";

test.beforeEach(({ page }) => failOnErrors(page));

test("lists the local replays and opens one from the picker", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.dashboard !== undefined);
  const picker = page.getByRole("dialog", { name: "Ouvrir un replay" });
  await expect(picker.getByRole("button", { name: /demo-4-voitures-spielberg/ })).toBeVisible();
  await picker.getByRole("button", { name: /demo-solo-spielberg/ }).click();
  await expect(picker).toBeHidden();
  await expect.poll(async () => (await state(page)).phase).toBe("ready");
  const ready = await state(page);
  expect(ready.carCount).toBe(1);
  expect(ready.visibleCars).toEqual([0]);
  expect(ready.map).toBe("Spielberg");
  expect(new URL(page.url()).searchParams.get("replay")).toBe("demo-solo-spielberg.npz");
});

test("draws the circuit and four cars with the Aerial camera", async ({ page }) => {
  await openReplay(page, "demo-4-voitures-spielberg");
  const ready = await state(page);
  expect(ready.camera).toBe("Aérienne");
  expect(ready.mapAvailable).toBe(true);
  expect(ready.visibleCars).toEqual([0, 1, 2, 3]);
  // Ground, asphalt, paint, walls from the occupancy image and DNF limit.
  expect(ready.triangles).toBeGreaterThan(50_000);
  expect(await renderedShare(page)).toBeGreaterThan(0.5);
  await expect(page.getByText("Spielberg · 4 voitures")).toBeVisible();
});

test("plays, pauses and reaches the end of the race", async ({ page }) => {
  await openReplay(page, "demo-4-voitures-spielberg");
  await page.getByRole("button", { name: "Lecture" }).click();
  await expect.poll(async () => (await state(page)).time).toBeGreaterThan(0.5);
  await page.getByRole("button", { name: "Pause" }).click();
  const paused = await state(page);
  expect(paused.playing).toBe(false);
  await page.waitForTimeout(300);
  expect((await state(page)).time).toBe(paused.time);

  await press(page, "End");
  const end = await state(page);
  expect(end.ended).toBe(true);
  expect(end.frame).toBe(end.frameCount - 1);
  // Every car has finished: the simulator has removed them from the track.
  expect(end.visibleCars).toEqual([]);

  await page.getByRole("button", { name: "Revoir depuis le début" }).click();
  await expect.poll(async () => (await state(page)).playing).toBe(true);
  expect((await state(page)).time).toBeLessThan(5);
});

test("seeks with the timeline and steps frame by frame", async ({ page }) => {
  await openReplay(page, "demo-4-voitures-spielberg");
  const slider = page.getByRole("slider", { name: "Position dans la course" });
  const box = await slider.boundingBox();
  if (!box) throw new Error("no timeline");
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
  const middle = await state(page);
  expect(middle.frame).toBeGreaterThan(middle.frameCount * 0.48);
  expect(middle.frame).toBeLessThan(middle.frameCount * 0.52);
  expect(middle.visibleCars).toEqual([0, 1, 2, 3]);

  await press(page, "ArrowRight");
  expect((await state(page)).frame).toBe(middle.frame + 1);
  await press(page, "Shift+ArrowLeft");
  expect((await state(page)).frame).toBe(middle.frame - 19);

  await page.getByRole("button", { name: "4×" }).click();
  expect((await state(page)).speed).toBe(4);
});

test("hides a car from the frame where it is no longer active", async ({ page }) => {
  await openReplay(page, "demo-dnf-sochi");
  expect((await state(page)).visibleCars).toEqual([0, 1, 2, 3]);
  await press(page, "End", "Shift+ArrowLeft", "Shift+ArrowLeft");
  // Near the end, the two cars that had a DNF are gone and the two others still race.
  expect((await state(page)).visibleCars).toEqual([0, 2]);
});

test("zooms, moves and resets the Aerial camera", async ({ page }) => {
  await openReplay(page, "demo-solo-spielberg");
  const home = (await state(page)).view;
  await page.mouse.move(640, 400);
  await page.mouse.wheel(0, -600);
  await expect
    .poll(async () => (await state(page)).view.pixelsPerMetre)
    .toBeGreaterThan(home.pixelsPerMetre * 1.2);
  await page.mouse.down();
  await page.mouse.move(540, 300, { steps: 5 });
  await page.mouse.up();
  await expect.poll(async () => (await state(page)).view.x).toBeGreaterThan(home.x + 1);
  // Dragging the ground up moves the view to the south.
  expect((await state(page)).view.y).toBeLessThan(home.y - 1);

  await press(page, "r");
  const reset = (await state(page)).view;
  expect(reset.pixelsPerMetre).toBeCloseTo(home.pixelsPerMetre, 3);
  expect(reset.x).toBeCloseTo(home.x, 3);
});

test("shows and hides the DNF limit", async ({ page }) => {
  await openReplay(page, "demo-solo-spielberg");
  const toggle = page.getByRole("switch", { name: "Limite de DNF" });
  await expect(toggle).toHaveAttribute("aria-checked", "true");
  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-checked", "false");
  expect((await state(page)).limitVisible).toBe(false);
});

test("explains why an invalid file is refused and stays usable", async ({ page }) => {
  await page.goto("/");
  await page.waitForFunction(() => window.dashboard !== undefined);
  await page.locator('input[type="file"]').setInputFiles({
    name: "notes.npz",
    mimeType: "application/octet-stream",
    buffer: Buffer.from("this is not a replay"),
  });
  const alert = page.getByRole("alert");
  await expect(alert).toContainText("Fichier de replay invalide.");
  await expect(alert).toContainText("archive NPZ");
  expect((await state(page)).phase).toBe("library");

  // The picker still works after the error.
  await page.getByRole("button", { name: /demo-solo-spielberg/ }).click();
  await expect.poll(async () => (await state(page)).phase).toBe("ready");
});

test("opens a replay whose map is missing, without the circuit", async ({ page }) => {
  await openReplay(page, "unknown-map");
  const ready = await state(page);
  expect(ready.mapAvailable).toBe(false);
  expect(ready.visibleCars).toEqual([0, 1]);
  await expect(page.getByRole("status")).toContainText("Carte « Atlantis » introuvable");
  await expect(page.getByRole("switch", { name: "Limite de DNF" })).toBeDisabled();
  expect(await renderedShare(page)).toBeGreaterThan(0);
});

test("opens a replay of an old recorder that only has poses and statuses", async ({ page }) => {
  await openReplay(page, "old-format");
  const ready = await state(page);
  expect(ready.carCount).toBe(2);
  expect(ready.map).toBeNull();
  await expect(page.getByRole("status").first()).toContainText("ne nomme pas sa carte");
});

test("reports a replay that the server does not have", async ({ page }) => {
  await page.goto("/?replay=absent.npz");
  await expect(page.getByRole("alert")).toContainText("Replay introuvable.");
});
