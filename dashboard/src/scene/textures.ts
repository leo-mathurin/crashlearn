/**
 * Procedural surface textures (asphalt, grass, concrete), drawn once on a canvas with a seeded
 * generator. No image file is downloaded: the dashboard works offline and the result is the
 * same on every machine.
 */
import * as THREE from "three/webgpu";

const SIZE = 512;

/** Small seeded generator (mulberry32): the textures are identical at each load. */
function random(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Tileable value noise: sum of `octaves` layers on a grid that wraps at the borders. */
function tileableNoise(seed: number, baseCells: number, octaves: number): Float32Array {
  const field = new Float32Array(SIZE * SIZE);
  let amplitude = 1;
  let total = 0;
  for (let octave = 0; octave < octaves; octave++) {
    const cells = baseCells << octave;
    const next = random(seed + octave * 101);
    const grid = Float32Array.from({ length: cells * cells }, next);
    const scale = cells / SIZE;
    for (let y = 0; y < SIZE; y++) {
      const gy = y * scale;
      const y0 = Math.floor(gy);
      const fy = gy - y0;
      const sy = fy * fy * (3 - 2 * fy);
      for (let x = 0; x < SIZE; x++) {
        const gx = x * scale;
        const x0 = Math.floor(gx);
        const fx = gx - x0;
        const sx = fx * fx * (3 - 2 * fx);
        const x1 = (x0 + 1) % cells;
        const y1 = (y0 + 1) % cells;
        const top = (grid[y0 * cells + x0] ?? 0) * (1 - sx) + (grid[y0 * cells + x1] ?? 0) * sx;
        const bottom = (grid[y1 * cells + x0] ?? 0) * (1 - sx) + (grid[y1 * cells + x1] ?? 0) * sx;
        const index = y * SIZE + x;
        field[index] = (field[index] ?? 0) + (top * (1 - sy) + bottom * sy) * amplitude;
      }
    }
    total += amplitude;
    amplitude *= 0.5;
  }
  for (let i = 0; i < field.length; i++) field[i] = (field[i] ?? 0) / total;
  return field;
}

function canvasOf(pixels: Uint8ClampedArray<ArrayBuffer>): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = SIZE;
  canvas.height = SIZE;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("2D canvas is not available");
  context.putImageData(new ImageData(pixels, SIZE, SIZE), 0, 0);
  return canvas;
}

function colorTexture(pixels: Uint8ClampedArray<ArrayBuffer>, anisotropy: number): THREE.Texture {
  const texture = new THREE.CanvasTexture(canvasOf(pixels));
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.anisotropy = anisotropy;
  return texture;
}

/** Tangent-space normal map from a height field that wraps at the borders. */
function normalTexture(height: Float32Array, strength: number, anisotropy: number): THREE.Texture {
  const pixels = new Uint8ClampedArray(SIZE * SIZE * 4);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const left = height[y * SIZE + ((x - 1 + SIZE) % SIZE)] ?? 0;
      const right = height[y * SIZE + ((x + 1) % SIZE)] ?? 0;
      const up = height[((y - 1 + SIZE) % SIZE) * SIZE + x] ?? 0;
      const down = height[((y + 1) % SIZE) * SIZE + x] ?? 0;
      const nx = (left - right) * strength;
      const ny = (down - up) * strength;
      const length = Math.hypot(nx, ny, 1);
      const index = (y * SIZE + x) * 4;
      pixels[index] = (nx / length) * 127.5 + 127.5;
      pixels[index + 1] = (ny / length) * 127.5 + 127.5;
      pixels[index + 2] = (1 / length) * 127.5 + 127.5;
      pixels[index + 3] = 255;
    }
  }
  const texture = new THREE.CanvasTexture(canvasOf(pixels));
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.anisotropy = anisotropy;
  return texture;
}

export interface Surface {
  map: THREE.Texture;
  normalMap: THREE.Texture;
}

/** Asphalt: dark binder with pale aggregate stones and slow tone variations. */
export function asphalt(anisotropy: number): Surface {
  const grain = tileableNoise(11, 64, 3);
  const blotches = tileableNoise(12, 4, 3);
  const next = random(13);
  const height = new Float32Array(SIZE * SIZE);
  const pixels = new Uint8ClampedArray(SIZE * SIZE * 4);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const stone = next() > 0.93 ? 0.5 + next() * 0.5 : 0;
    const tone = 0.62 + (grain[i] ?? 0) * 0.5 + ((blotches[i] ?? 0) - 0.5) * 0.35 + stone * 0.5;
    height[i] = (grain[i] ?? 0) + stone * 0.8;
    pixels[i * 4] = 44 * tone;
    pixels[i * 4 + 1] = 46 * tone;
    pixels[i * 4 + 2] = 51 * tone;
    pixels[i * 4 + 3] = 255;
  }
  return {
    map: colorTexture(pixels, anisotropy),
    normalMap: normalTexture(height, 2.2, anisotropy),
  };
}

/** Grass: mottled greens with fine blade-scale grain. */
export function grass(anisotropy: number): Surface {
  const patches = tileableNoise(21, 4, 4);
  const blades = tileableNoise(22, 128, 2);
  const dry = tileableNoise(23, 8, 2);
  const height = new Float32Array(SIZE * SIZE);
  const pixels = new Uint8ClampedArray(SIZE * SIZE * 4);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const patch = patches[i] ?? 0;
    const blade = blades[i] ?? 0;
    const dryness = Math.max(0, (dry[i] ?? 0) - 0.55) * 1.6;
    const light = 0.72 + patch * 0.3 + (blade - 0.5) * 0.45;
    height[i] = blade;
    pixels[i * 4] = (58 + dryness * 34) * light;
    pixels[i * 4 + 1] = (92 + dryness * 14) * light;
    pixels[i * 4 + 2] = (44 + dryness * 4) * light;
    pixels[i * 4 + 3] = 255;
  }
  return {
    map: colorTexture(pixels, anisotropy),
    normalMap: normalTexture(height, 3, anisotropy),
  };
}

/** Cast concrete: pale gray with pores and soft stains. */
export function concrete(anisotropy: number): Surface {
  const stains = tileableNoise(31, 4, 4);
  const pores = tileableNoise(32, 96, 2);
  const next = random(33);
  const height = new Float32Array(SIZE * SIZE);
  const pixels = new Uint8ClampedArray(SIZE * SIZE * 4);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const pore = next() > 0.985 ? 0.45 : 0;
    const tone = 0.78 + (stains[i] ?? 0) * 0.3 + ((pores[i] ?? 0) - 0.5) * 0.16 - pore;
    height[i] = (pores[i] ?? 0) * 0.6 - pore;
    pixels[i * 4] = 214 * tone;
    pixels[i * 4 + 1] = 212 * tone;
    pixels[i * 4 + 2] = 205 * tone;
    pixels[i * 4 + 3] = 255;
  }
  return {
    map: colorTexture(pixels, anisotropy),
    normalMap: normalTexture(height, 1.6, anisotropy),
  };
}
