/**
 * Local server of the race dashboard. It serves the page (bundled by Bun), the replays of a
 * local folder and the map assets of the vendored simulator. No remote service is necessary.
 *
 *   bun server.ts [--port 5173] [--replays ../replays] [--maps ../vendor/simulation/maps] [--dev]
 */
import { readdir, stat } from "node:fs/promises";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";

import index from "./index.html";

const { values } = parseArgs({
  args: Bun.argv.slice(2),
  options: {
    port: { type: "string", default: process.env.PORT ?? "5173" },
    replays: { type: "string", default: resolve(import.meta.dir, "../replays") },
    maps: { type: "string", default: resolve(import.meta.dir, "../vendor/simulation/maps") },
    dev: { type: "boolean", default: false },
  },
});

const replaysDir = resolve(values.replays);
const mapsDir = resolve(values.maps);
const SAFE_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

/** Map asset names of the API, and the file each one has in a map folder of the simulator. */
const MAP_ASSETS: Record<string, (name: string) => string> = {
  "map.yaml": (name) => `${name}_map.yaml`,
  "map.png": (name) => `${name}_map.png`,
  "centerline.csv": (name) => `${name}_centerline.csv`,
};

function notFound(message: string): Response {
  return Response.json({ error: message }, { status: 404 });
}

async function listReplays(): Promise<Response> {
  let names: string[];
  try {
    names = await readdir(replaysDir);
  } catch {
    return Response.json({ directory: replaysDir, replays: [] });
  }
  const files = names.filter((name) => name.endsWith(".npz") && SAFE_NAME.test(name));
  const replays = await Promise.all(
    files.map(async (name) => {
      const info = await stat(join(replaysDir, name));
      return { name, size: info.size, modified: info.mtimeMs };
    }),
  );
  replays.sort((a, b) => b.modified - a.modified);
  return Response.json({ directory: replaysDir, replays });
}

async function serveFile(path: string, type: string): Promise<Response> {
  const file = Bun.file(path);
  if (!(await file.exists())) return notFound("not found");
  return new Response(file, { headers: { "content-type": type, "cache-control": "no-cache" } });
}

const server = Bun.serve({
  port: Number(values.port),
  development: values.dev,
  routes: {
    "/": index,
    "/api/replays": () => listReplays(),
    "/api/replays/:name": (request) => {
      const { name } = request.params;
      if (!SAFE_NAME.test(name) || !name.endsWith(".npz")) return notFound("invalid replay name");
      return serveFile(join(replaysDir, name), "application/octet-stream");
    },
    "/api/maps/:name/:asset": (request) => {
      const { name, asset } = request.params;
      const fileOf = MAP_ASSETS[asset];
      if (!SAFE_NAME.test(name) || !fileOf) return notFound("invalid map asset");
      // The default map of the simulator is in `examples/` with the prefix `example`.
      const folder = name === "example" ? "examples" : name;
      const type = asset.endsWith(".png") ? "image/png" : "text/plain; charset=utf-8";
      return serveFile(join(mapsDir, folder, fileOf(name)), type);
    },
  },
  fetch: () => notFound("not found"),
});

console.info(`Dashboard de course : ${server.url}`);
console.info(`Replays : ${replaysDir}`);
console.info(`Cartes  : ${mapsDir}`);
