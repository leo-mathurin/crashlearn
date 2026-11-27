import { expect, test } from "@playwright/test";

import { failOnErrors, openReplay, press } from "./dashboard";

test.beforeEach(({ page }) => failOnErrors(page));

const SIZES = [
  [1920, 1080],
  [1280, 720],
  [1000, 640],
  [1000, 390],
  [768, 1024],
  [390, 844],
  [844, 390],
] as const;

for (const [width, height] of SIZES) {
  test(`keeps the panels apart and on screen at ${width} x ${height}`, async ({ page }) => {
    await page.setViewportSize({ width, height });
    await openReplay(page, "demo-dnf-sochi");
    // The Chase camera shows the most controls: its menu « Suivre » is in the top bar.
    await press(page, "c");
    const problems = await page.evaluate(() => {
      const panels = [...document.querySelectorAll<HTMLElement>("#hud .panel")]
        .filter((panel) => panel.getClientRects().length > 0)
        .map((panel) => ({ name: panel.className, box: panel.getBoundingClientRect() }));
      const found: string[] = [];
      panels.forEach((a, index) => {
        if (a.box.left < 0 || a.box.top < 0) found.push(`${a.name} is off screen`);
        if (a.box.right > innerWidth + 0.5 || a.box.bottom > innerHeight + 0.5) {
          found.push(`${a.name} is off screen`);
        }
        for (const b of panels.slice(index + 1)) {
          const apart =
            a.box.right <= b.box.left + 0.5 ||
            b.box.right <= a.box.left + 0.5 ||
            a.box.bottom <= b.box.top + 0.5 ||
            b.box.bottom <= a.box.top + 0.5;
          if (!apart) found.push(`${a.name} is over ${b.name}`);
        }
      });
      return found;
    });
    expect(problems).toEqual([]);
    await expect(page.getByRole("button", { name: "Ouvrir" })).toBeVisible();
    await expect(page.getByRole("radio", { name: "Libre" })).toBeVisible();
  });
}
