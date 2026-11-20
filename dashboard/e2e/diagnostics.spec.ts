import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { expect, test } from "@playwright/test";

import { failOnErrors, openReplay, press, state } from "./dashboard";

interface CarReference {
  status: number;
  rank: number;
  laps: number;
  progress: number;
  speed: number;
  targetSpeed: number;
  steering: number;
  targetSteering: number;
}

function reference(): Record<string, CarReference[]> {
  const here = dirname(fileURLToPath(import.meta.url));
  return JSON.parse(
    readFileSync(join(here, "../.e2e-replays/demo-dnf-sochi.expected.json"), "utf8"),
  );
}

test.beforeEach(({ page }) => failOnErrors(page));

test("lists the cars in race order with name, lap and progress", async ({ page }) => {
  await openReplay(page, "demo-dnf-sochi");
  await press(page, ...Array.from({ length: 10 }, () => "Shift+ArrowRight"));
  const now = await state(page);
  expect(now.frame).toBe(200);
  const cars = reference()["200"] ?? [];
  expect(now.order).toEqual(
    cars.map((_, car) => car).toSorted((a, b) => (cars[a]?.rank ?? 0) - (cars[b]?.rank ?? 0)),
  );
  const rows = page.getByRole("listitem");
  await expect(rows).toHaveCount(4);
  const names = ["1-driver1", "2-driver2", "3-driver1b", "4-crash"];
  for (const [car, name] of names.entries()) {
    const row = rows.filter({ hasText: name });
    const expected = cars[car];
    if (!expected) throw new Error("no reference");
    await expect(row.locator(".row-position")).toHaveText(String(expected.rank));
    await expect(row.locator(".row-detail")).toHaveText(
      `Tour ${expected.laps + 1} · ${Math.floor(expected.progress * 100)} %`,
    );
  }
});

test("shows the recorded speed, steering and commands of the selected car", async ({ page }) => {
  await openReplay(page, "demo-dnf-sochi");
  const panel = page.locator(".telemetry");
  await expect(panel.locator(".telemetry-name")).toHaveText("1-driver1");
  await press(page, ...Array.from({ length: 10 }, () => "Shift+ArrowRight"));

  for (const car of [0, 1, 2]) {
    await press(page, String(car + 1));
    const expected = reference()["200"]?.[car];
    const shown = (await state(page)).telemetry;
    if (!expected || !shown) throw new Error("no telemetry");
    expect((await state(page)).selectedCar).toBe(car);
    expect(shown.speed).toBeCloseTo(expected.speed, 5);
    expect(shown.targetSpeed).toBeCloseTo(expected.targetSpeed, 5);
    expect(shown.steering).toBeCloseTo(expected.steering, 5);
    expect(shown.targetSteering).toBeCloseTo(expected.targetSteering, 5);
    const speedText = expected.speed.toFixed(2).replace(".", ",");
    await expect(panel.locator(".gauge-value").first()).toHaveText(speedText);
    await expect(panel.locator(".gauge-target").first()).toHaveText(
      `Consigne ${expected.targetSpeed.toFixed(2).replace(".", ",")}`,
    );
    const degrees = ((expected.steering * 180) / Math.PI).toFixed(1).replace(".", ",");
    await expect(panel.locator(".gauge-value").nth(1)).toHaveText(`${degrees.replace("-", "−")}°`);
  }
});

test("selects a car from the leaderboard", async ({ page }) => {
  await openReplay(page, "demo-dnf-sochi");
  await page.getByRole("listitem").filter({ hasText: "3-driver1b" }).click();
  expect((await state(page)).selectedCar).toBe(2);
  await expect(page.locator(".telemetry-name")).toHaveText("3-driver1b");
  await expect(page.getByRole("listitem").filter({ hasText: "3-driver1b" })).toHaveAttribute(
    "aria-current",
    "true",
  );
});

test("draws the LiDAR of the selected car and hides it on demand", async ({ page }) => {
  await openReplay(page, "demo-dnf-sochi");
  await press(page, "Shift+ArrowRight");
  expect((await state(page)).lidarVisible).toBe(true);
  const toggle = page.getByRole("switch", { name: "LiDAR" });
  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-checked", "false");
  await expect.poll(async () => (await state(page)).lidarVisible).toBe(false);
  await press(page, "i");
  await expect.poll(async () => (await state(page)).lidarVisible).toBe(true);
});

test("marks a DNF at the last pose, from the frame of the DNF, and not an arrival", async ({
  page,
}) => {
  await openReplay(page, "demo-dnf-sochi");
  expect((await state(page)).dnfMarkers).toEqual([]);
  await expect(page.locator(".dnf-tag:visible")).toHaveCount(0);

  await press(page, "End");
  const end = await state(page);
  // Cars 1 and 3 ended in DNF; cars 0 and 2 finished and get no marker.
  expect(end.dnfMarkers).toEqual([1, 3]);
  expect(end.visibleCars).toEqual([]);
  await expect(page.locator(".dnf-tag:visible")).toHaveText(["DNF2-driver2", "DNF4-crash"]);

  // The inactive selected car has no LiDAR, and its panel says how the race ended.
  await press(page, "4");
  expect((await state(page)).lidarVisible).toBe(false);
  await expect(page.locator(".status-pill")).toHaveText("DNF");
});

test("shows the final classification at the last frame", async ({ page }) => {
  await openReplay(page, "demo-dnf-sochi");
  await expect(page.locator(".leaderboard h2")).toHaveText("Classement");
  await press(page, "End");
  await expect(page.locator(".leaderboard h2")).toHaveText("Classement final");
  const cars = reference()[String((await state(page)).frameCount - 1)] ?? [];
  const rows = page.getByRole("listitem");
  for (const [car, name] of ["1-driver1", "2-driver2", "3-driver1b", "4-crash"].entries()) {
    const row = rows.filter({ hasText: name });
    await expect(row.locator(".row-position")).toHaveText(String(cars[car]?.rank));
    await expect(row.locator(".row-detail")).toHaveText(
      cars[car]?.status === 2 ? /^Arrivée · \d:\d\d\.\d$/ : /^DNF · \d:\d\d\.\d$/,
    );
  }
});

test("says that a value is unavailable when an old replay does not have it", async ({ page }) => {
  await openReplay(page, "old-format");
  await expect(page.getByRole("status").filter({ hasText: "LiDAR" })).toContainText(
    "n'a pas de mesures LiDAR",
  );
  await expect(page.getByRole("switch", { name: "LiDAR" })).toBeDisabled();
  await expect(page.getByRole("listitem").first().locator(".row-detail")).toHaveText(
    "indisponible",
  );
  const panel = page.locator(".telemetry");
  await expect(panel.locator(".gauge-value").first()).toHaveText("–");
  await expect(panel.locator(".chart-note")).toHaveText("indisponible");
  expect((await state(page)).telemetry?.speed).toBeNull();
});
