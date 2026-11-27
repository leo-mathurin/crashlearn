/**
 * Decoder for NumPy `.npz` archives (a ZIP of `.npy` arrays), as written by
 * `np.savez_compressed` in `vendor/simulation/sim_recorder.py`.
 */
import { unzipSync } from "fflate";

export type NumericArray = Float32Array | Float64Array | Int32Array | Uint8Array;

export interface NpyArray {
  /** NumPy dtype string, for example `<f4`. */
  dtype: string;
  shape: number[];
  /** Numeric data in C order. Empty for string arrays. */
  data: NumericArray;
  /** Decoded items of a unicode (`<U`) array. Empty for numeric arrays. */
  strings: string[];
}

export class NpzError extends Error {
  override name = "NpzError";
}

const MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];

function parseHeader(text: string): { descr: string; fortran: boolean; shape: number[] } {
  const descr = /'descr':\s*'([^']+)'/.exec(text)?.[1];
  const fortran = /'fortran_order':\s*(True|False)/.exec(text)?.[1];
  const shape = /'shape':\s*\(([^)]*)\)/.exec(text)?.[1];
  if (descr === undefined || fortran === undefined || shape === undefined) {
    throw new NpzError(`unreadable .npy header: ${text.trim()}`);
  }
  const dims = shape
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part !== "")
    .map(Number);
  if (dims.some((dim) => !Number.isInteger(dim) || dim < 0)) {
    throw new NpzError(`invalid shape in .npy header: (${shape})`);
  }
  return { descr, fortran: fortran === "True", shape: dims };
}

function decodeNumeric(descr: string, bytes: Uint8Array, count: number): NumericArray {
  const kind = descr.slice(1);
  if (descr.startsWith(">") && kind !== "b1" && kind !== "u1" && kind !== "i1") {
    throw new NpzError(`big-endian dtype is not supported: ${descr}`);
  }
  const sizes: Record<string, number> = { f4: 4, f8: 8, i4: 4, i8: 8, u1: 1, b1: 1, i2: 2, u2: 2 };
  const size = sizes[kind];
  if (size === undefined) throw new NpzError(`unsupported dtype: ${descr}`);
  if (bytes.byteLength < count * size) {
    throw new NpzError(`truncated array: ${bytes.byteLength} bytes for ${count} items of ${descr}`);
  }
  // Copy to a new buffer: the offset in the archive is not always aligned for a typed array.
  const buffer = bytes.slice(0, count * size).buffer;
  switch (kind) {
    case "f4":
      return new Float32Array(buffer);
    case "f8":
      return new Float64Array(buffer);
    case "i4":
      return new Int32Array(buffer);
    case "u1":
    case "b1":
      return new Uint8Array(buffer);
    case "i2":
      return Int32Array.from(new Int16Array(buffer));
    case "u2":
      return Int32Array.from(new Uint16Array(buffer));
    default:
      // i8: replay counters are far below 2^53, a float64 holds them exactly.
      return Float64Array.from(new BigInt64Array(buffer), Number);
  }
}

function decodeUnicode(descr: string, bytes: Uint8Array, count: number): string[] {
  const width = Number(descr.slice(2));
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.byteLength < count * width * 4) throw new NpzError("truncated string array");
  const strings: string[] = [];
  for (let item = 0; item < count; item++) {
    let text = "";
    for (let char = 0; char < width; char++) {
      const code = view.getUint32((item * width + char) * 4, true);
      if (code === 0) break;
      text += String.fromCodePoint(code);
    }
    strings.push(text);
  }
  return strings;
}

export function parseNpy(bytes: Uint8Array): NpyArray {
  if (bytes.length < 10 || MAGIC.some((byte, index) => bytes[index] !== byte)) {
    throw new NpzError("not a .npy array (bad magic number)");
  }
  const major = bytes[6];
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerLength = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerStart = major === 1 ? 10 : 12;
  const dataStart = headerStart + headerLength;
  if (dataStart > bytes.length) throw new NpzError("truncated .npy header");
  const header = parseHeader(new TextDecoder().decode(bytes.subarray(headerStart, dataStart)));
  if (header.fortran && header.shape.length > 1) {
    throw new NpzError("Fortran-ordered arrays are not supported");
  }
  const count = header.shape.reduce((product, dim) => product * dim, 1);
  const body = bytes.subarray(dataStart);
  if (/^[<=|]U\d+$/.test(header.descr)) {
    return {
      dtype: header.descr,
      shape: header.shape,
      data: new Uint8Array(0),
      strings: decodeUnicode(header.descr, body, count),
    };
  }
  return {
    dtype: header.descr,
    shape: header.shape,
    data: decodeNumeric(header.descr, body, count),
    strings: [],
  };
}

/** Decodes all the arrays of an `.npz` archive, by array name. */
export function parseNpz(bytes: Uint8Array): Map<string, NpyArray> {
  let files: Record<string, Uint8Array>;
  try {
    files = unzipSync(bytes);
  } catch (error) {
    throw new NpzError(`not a ZIP archive: ${error instanceof Error ? error.message : error}`);
  }
  const arrays = new Map<string, NpyArray>();
  for (const [name, content] of Object.entries(files)) {
    if (!name.endsWith(".npy")) continue;
    arrays.set(name.slice(0, -4), parseNpy(content));
  }
  if (arrays.size === 0) throw new NpzError("the archive contains no .npy array");
  return arrays;
}
