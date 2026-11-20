/** Loads the assets of a map from the local server: metadata, occupancy image, centerline. */
import { type Centerline, type MapMeta, parseCenterline, parseMapYaml } from "./track";

export interface OccupancyImage {
  /** One gray value per pixel, row 0 at the top of the image. */
  gray: Uint8Array;
  width: number;
  height: number;
}

export interface TrackAssets {
  name: string;
  meta: MapMeta;
  image: OccupancyImage;
  centerline: Centerline;
}

export class MapNotFoundError extends Error {
  override name = "MapNotFoundError";
  constructor(readonly mapName: string) {
    super(`map not found: ${mapName}`);
  }
}

async function fetchAsset(name: string, asset: string): Promise<Response> {
  const response = await fetch(`/api/maps/${encodeURIComponent(name)}/${asset}`);
  if (response.status === 404) throw new MapNotFoundError(name);
  if (!response.ok) throw new Error(`map asset ${asset}: HTTP ${response.status}`);
  return response;
}

async function decodeGray(blob: Blob): Promise<OccupancyImage> {
  // No colour conversion: the gray levels must be the ones that the simulator reads.
  const bitmap = await createImageBitmap(blob, {
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  });
  const { width, height } = bitmap;
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("2D canvas is not available");
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  const rgba = context.getImageData(0, 0, width, height).data;
  const gray = new Uint8Array(width * height);
  for (let i = 0; i < gray.length; i++) gray[i] = rgba[i * 4] ?? 255;
  return { gray, width, height };
}

export async function loadTrackAssets(name: string): Promise<TrackAssets> {
  const [yaml, png, csv] = await Promise.all([
    fetchAsset(name, "map.yaml"),
    fetchAsset(name, "map.png"),
    fetchAsset(name, "centerline.csv"),
  ]);
  return {
    name,
    meta: parseMapYaml(await yaml.text()),
    image: await decodeGray(await png.blob()),
    centerline: parseCenterline(await csv.text()),
  };
}
