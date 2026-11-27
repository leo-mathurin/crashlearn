import { defineConfig, devices } from "@playwright/test";

const PORT = 5187;

/**
 * Browser tests of the dashboard on Chromium and Firefox. `e2e/setup.ts` prepares the replays:
 * the demo races of `scripts/record_replay.py` and variants with a missing map or old arrays.
 */
export default defineConfig({
  testDir: "e2e",
  globalSetup: "./e2e/setup.ts",
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  reporter: [["list"]],
  // Each page builds a 3D scene, with software rendering on a machine that has no GPU.
  workers: 4,
  timeout: 120_000,
  expect: { timeout: 15_000 },
  use: { baseURL: `http://localhost:${PORT}`, viewport: { width: 1280, height: 800 } },
  webServer: {
    command: `bun server.ts --port ${PORT} --replays .e2e-replays`,
    url: `http://localhost:${PORT}/api/replays`,
    reuseExistingServer: !process.env.CI,
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "firefox", use: { ...devices["Desktop Firefox"] } },
    // WebGPU needs a GPU and, on Linux, these flags and a window: `E2E_WEBGPU=1` adds the project.
    ...(process.env.E2E_WEBGPU
      ? [
          {
            name: "chromium-webgpu",
            use: {
              ...devices["Desktop Chrome"],
              headless: false,
              launchOptions: {
                args: [
                  "--enable-unsafe-webgpu",
                  "--enable-features=Vulkan,VulkanFromANGLE",
                  "--use-angle=vulkan",
                  "--ignore-gpu-blocklist",
                ],
              },
            },
          },
        ]
      : []),
  ],
});
