/**
 * Prepares the replays of the browser tests in `.e2e-replays/`: the demo races, recorded with
 * the real simulator if they are not there yet, and variants of the small fixture.
 */
import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { unzipSync, zipSync } from "fflate";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "../..");
const target = join(here, "../.e2e-replays");

export const DEMOS = ["demo-solo-spielberg", "demo-4-voitures-spielberg", "demo-dnf-sochi"];

/** A zero-dimensional NumPy unicode array, as `np.str_` is saved. */
function unicodeScalar(text: string): Uint8Array {
  let header = `{'descr': '<U${text.length}', 'fortran_order': False, 'shape': (), }`;
  header += `${" ".repeat((64 - ((10 + header.length + 1) % 64)) % 64)}\n`;
  const bytes = new Uint8Array(10 + header.length + text.length * 4);
  bytes.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, header.length & 255, header.length >> 8]);
  bytes.set(new TextEncoder().encode(header), 10);
  const view = new DataView(bytes.buffer);
  [...text].forEach((char, index) => {
    view.setUint32(10 + header.length + index * 4, char.codePointAt(0) ?? 0, true);
  });
  return bytes;
}

export default function setup(): void {
  mkdirSync(target, { recursive: true });
  if (DEMOS.some((name) => !existsSync(join(root, "replays", `${name}.npz`)))) {
    execFileSync("uv", ["run", "python", "scripts/record_replay.py", "--demo"], {
      cwd: root,
      stdio: "inherit",
    });
  }
  for (const name of DEMOS) {
    copyFileSync(join(root, "replays", `${name}.npz`), join(target, `${name}.npz`));
  }

  // Values of one demo race read with NumPy: the tests compare the page with them.
  const dump = `
import json, sys, numpy as np
ep = np.load(sys.argv[1])
frames = [0, 200, ep["poses"].shape[0] - 1]
print(json.dumps({str(f): [{
    "status": int(ep["status"][f, c]), "rank": int(ep["ranks"][f, c]),
    "laps": int(ep["lap_count"][f, c]), "progress": float(ep["progress"][f, c]),
    "speed": float(ep["velocity"][f, c]), "targetSpeed": float(ep["actions"][f, c, 0]),
    "steering": float(ep["steering"][f, c]), "targetSteering": float(ep["actions"][f, c, 1]),
} for c in range(int(ep["num_cars"]))] for f in frames}))`;
  writeFileSync(
    join(target, "demo-dnf-sochi.expected.json"),
    execFileSync("uv", ["run", "python", "-c", dump, join(target, "demo-dnf-sochi.npz")], {
      cwd: root,
    }),
  );

  const fixture = unzipSync(readFileSync(join(here, "../tests/fixtures/two-cars-40-frames.npz")));
  writeFileSync(
    join(target, "unknown-map.npz"),
    zipSync({ ...fixture, "map_name.npy": unicodeScalar("Atlantis") }),
  );
  const { "poses.npy": poses, "status.npy": status, "sim_dt.npy": simDt } = fixture;
  if (!poses || !status || !simDt) throw new Error("the fixture replay is incomplete");
  writeFileSync(
    join(target, "old-format.npz"),
    zipSync({ "poses.npy": poses, "status.npy": status, "sim_dt.npy": simDt }),
  );
  writeFileSync(join(target, "not-a-replay.npz"), "this is not a replay\n");
}
