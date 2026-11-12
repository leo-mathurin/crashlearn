/**
 * Measures the fluidity of the dashboard on a real replay, in Chromium (WebGPU and WebGL 2)
 * and Firefox, for each camera, with and without the scenery.
 *
 *   bun start --port 5173                      # in another terminal
 *   bun scripts/measure-performance.ts [replay name] [port] [width x height, 1920x1080]
 *
 * On Linux without a screen: `xvfb-run -a bun scripts/measure-performance.ts`.
 * Three numbers per case: the frame rate of a real playback, which the screen refresh limits;
 * the main-thread time of one frame; and the cost of one frame without the refresh limit, the
 * GPU work included (`window.dashboard.benchmark`).
 */
import { type Browser, chromium, firefox } from "@playwright/test";

const replay = process.argv[2] ?? "demo-4-voitures-spielberg";
const port = process.argv[3] ?? "5173";
const [width = 1920, height = 1080] = (process.argv[4] ?? "").split("x").map(Number);
const GPU_FLAGS = [
  "--enable-unsafe-webgpu",
  "--enable-features=Vulkan,VulkanFromANGLE",
  "--use-angle=vulkan",
  "--ignore-gpu-blocklist",
];

const browsers: Record<string, { launch: () => Promise<Browser>; query: string }> = {
  "Chromium, WebGPU": {
    launch: () => chromium.launch({ headless: false, args: GPU_FLAGS }),
    query: "",
  },
  "Chromium, WebGL 2": {
    launch: () => chromium.launch({ headless: false, args: GPU_FLAGS }),
    query: "&renderer=webgl",
  },
  "Firefox, WebGL 2": { launch: () => firefox.launch({ headless: true }), query: "" },
};

console.info(
  `| Navigateur | Caméra | Décor | Lecture réelle | Calcul par frame | Coût d'une frame | Triangles |`,
);
console.info(`| --- | --- | --- | ---: | ---: | ---: | ---: |`);
for (const [label, { launch, query }] of Object.entries(browsers)) {
  const browser = await launch();
  const page = await browser.newPage({ viewport: { width, height } });
  await page.goto(`http://localhost:${port}/?replay=${replay}.npz${query}`);
  await page.waitForFunction(() => window.dashboard?.state().phase === "ready", null, {
    timeout: 60_000,
  });
  const backend = await page.evaluate(() => window.dashboard?.state().backend);
  // One minute into the race: the cars are spread on the circuit.
  for (let i = 0; i < 60; i++) await page.keyboard.press("Shift+ArrowRight");
  for (const camera of ["aerial", "free", "chase"] as const) {
    for (const decor of [true, false]) {
      const result = await page.evaluate(
        async ([mode, scenery]) => {
          const dashboard = window.dashboard;
          if (!dashboard) throw new Error("the dashboard did not start");
          dashboard.setDecorVisible(scenery);
          dashboard.setCamera(mode, false);
          window.dispatchEvent(new KeyboardEvent("keydown", { key: " " }));
          // The meter keeps the last 240 frames: 5 s of playback fill it.
          await new Promise((resolve) => setTimeout(resolve, 5000));
          const { perf, triangles, camera: name } = dashboard.state();
          window.dispatchEvent(new KeyboardEvent("keydown", { key: " " }));
          const bench = await dashboard.benchmark(300);
          return { perf, triangles, name, bench };
        },
        [camera, decor] as const,
      );
      const { perf, bench } = result;
      console.info(
        `| ${label} (${backend}) | ${result.name} | ${decor ? "oui" : "non"} | ` +
          `${perf.fps.toFixed(0)} fps, ${perf.frameMs.toFixed(1)} ms (p95 ${perf.frameP95Ms.toFixed(1)}) | ` +
          `${perf.workMs.toFixed(2)} ms | ${bench.msPerFrame.toFixed(2)} ms | ` +
          `${Math.round(result.triangles)} |`,
      );
    }
  }
  const memory = await page.evaluate(() => {
    const heap = (performance as { memory?: { usedJSHeapSize: number } }).memory;
    return heap ? heap.usedJSHeapSize / 1024 / 1024 : null;
  });
  if (memory !== null) console.info(`<!-- ${label}: tas JavaScript ${memory.toFixed(0)} Mo -->`);
  await browser.close();
}
