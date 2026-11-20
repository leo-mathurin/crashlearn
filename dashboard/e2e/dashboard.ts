import { expect, type Page } from "@playwright/test";

import type { DashboardState } from "../src/app";

export function state(page: Page): Promise<DashboardState> {
  return page.evaluate(() => {
    if (!window.dashboard) throw new Error("the dashboard did not start");
    return window.dashboard.state();
  });
}

/** Opens a replay of the test folder by its URL and waits for the scene. */
export async function openReplay(page: Page, name: string): Promise<void> {
  await page.goto(`/?replay=${name}.npz`);
  await page.waitForFunction(() => window.dashboard !== undefined);
  await expect.poll(async () => (await state(page)).phase, { timeout: 30_000 }).toBe("ready");
}

/** Sends keys, then waits until the frame loop has drawn their effect. */
export async function press(page: Page, ...keys: string[]): Promise<void> {
  for (const key of keys) await page.keyboard.press(key);
  await page.evaluate(
    () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      }),
  );
}

/**
 * Share of the canvas pixels that differ from the first one. A scene that did not render is
 * one flat colour: the share is then 0.
 */
export function renderedShare(page: Page): Promise<number> {
  return page.evaluate(
    () =>
      new Promise<number>((resolve) => {
        // Read in a frame callback: the drawing buffer is only valid until the frame is shown.
        requestAnimationFrame(() => {
          const scene = document.querySelector<HTMLCanvasElement>("#scene");
          if (!scene) throw new Error("no scene canvas");
          const copy = document.createElement("canvas");
          copy.width = 160;
          copy.height = 100;
          const context = copy.getContext("2d");
          if (!context) throw new Error("no 2D canvas");
          context.drawImage(scene, 0, 0, copy.width, copy.height);
          const { data } = context.getImageData(0, 0, copy.width, copy.height);
          let different = 0;
          for (let i = 4; i < data.length; i += 4) {
            if (data[i] !== data[0] || data[i + 1] !== data[1] || data[i + 2] !== data[2]) {
              different++;
            }
          }
          resolve(different / (data.length / 4));
        });
      }),
  );
}

/**
 * Fails the test on a page error or a console error. The browser also logs each HTTP 404 as a
 * console error: a missing map or replay is a case that the tests check in the page.
 */
export function failOnErrors(page: Page): void {
  page.on("pageerror", (error) => {
    throw error;
  });
  page.on("console", (message) => {
    if (message.text().startsWith("Failed to load resource")) return;
    if (message.type() === "error") throw new Error(`console error: ${message.text()}`);
  });
}
