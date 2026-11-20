/**
 * Replay model: the arrays of one recorded race, validated. The format is the one of
 * `vendor/simulation/sim_recorder.py`. All arrays are flat, in C order: frame, then car.
 */
import { type NpyArray, NpzError, parseNpz } from "./npz";

/** Car status values of the simulator (`agent_status`). */
export const STATUS = { DNF: 0, ACTIVE: 1, FINISHED: 2 } as const;

export const MAX_CARS = 4;

export interface Replay {
  /** File name of the replay. */
  name: string;
  frameCount: number;
  carCount: number;
  /** Simulated time between two frames, in seconds. */
  simDt: number;
  /** Simulated time of the last frame, in seconds. */
  duration: number;
  /** Map name recorded in the file, or null for a file that has none. */
  mapName: string | null;
  teamNames: string[];
  /** x, y (m) and heading (rad) for each frame and car. */
  poses: Float32Array;
  status: Int32Array;
  /** LiDAR ranges (m): frame, car, beam. */
  lidar: Float32Array | null;
  beamCount: number;
  velocity: Float32Array | null;
  steering: Float32Array | null;
  progress: Float32Array | null;
  lapCount: Int32Array | null;
  /** Commands of the driver: target speed (m/s) and steering (rad). */
  actions: Float32Array | null;
  ranks: Int32Array | null;
  friction: Float32Array | null;
  maxProgress: Float32Array | null;
  /** Names of the optional arrays that the file does not have. */
  missing: string[];
  /** For each car, the first frame where it is no longer ACTIVE (`frameCount` if none). */
  inactiveFrom: Int32Array;
}

export type ReplayErrorCode = "not-npz" | "missing-array" | "bad-shape" | "bad-value";

export class ReplayError extends Error {
  override name = "ReplayError";
  constructor(
    readonly code: ReplayErrorCode,
    /** Message for the user, in French. */
    message: string,
  ) {
    super(message);
  }
}

function numeric(arrays: Map<string, NpyArray>, key: string): NpyArray | null {
  const array = arrays.get(key);
  if (!array) return null;
  if (array.strings.length > 0 || array.dtype.includes("U")) {
    throw new ReplayError("bad-shape", `Le tableau « ${key} » devrait être numérique.`);
  }
  return array;
}

function expectShape(key: string, array: NpyArray, shape: number[]): void {
  const same =
    array.shape.length === shape.length && array.shape.every((dim, index) => dim === shape[index]);
  if (!same) {
    throw new ReplayError(
      "bad-shape",
      `Le tableau « ${key} » a la forme (${array.shape.join(", ")}) au lieu de (${shape.join(", ")}).`,
    );
  }
}

function asFloat32(array: NpyArray): Float32Array {
  return array.data instanceof Float32Array ? array.data : Float32Array.from(array.data);
}

function asInt32(array: NpyArray): Int32Array {
  return array.data instanceof Int32Array ? array.data : Int32Array.from(array.data);
}

/** Builds a replay from decoded arrays. Throws a `ReplayError` when the content is not valid. */
export function buildReplay(name: string, arrays: Map<string, NpyArray>): Replay {
  const poses = numeric(arrays, "poses");
  const status = numeric(arrays, "status");
  const simDtArray = numeric(arrays, "sim_dt");
  for (const [key, array] of [
    ["poses", poses],
    ["status", status],
    ["sim_dt", simDtArray],
  ] as const) {
    if (!array) {
      throw new ReplayError(
        "missing-array",
        `Le tableau « ${key} » est absent : ce fichier n'est pas un replay de course.`,
      );
    }
  }
  if (!poses || !status || !simDtArray) throw new Error("unreachable");

  if (poses.shape.length !== 3 || poses.shape[2] !== 3) {
    throw new ReplayError(
      "bad-shape",
      `Le tableau « poses » a la forme (${poses.shape.join(", ")}) au lieu de (frames, voitures, 3).`,
    );
  }
  const frameCount = poses.shape[0] ?? 0;
  const carCount = poses.shape[1] ?? 0;
  if (frameCount < 1) throw new ReplayError("bad-shape", "Le replay ne contient aucune frame.");
  if (carCount < 1 || carCount > MAX_CARS) {
    throw new ReplayError(
      "bad-shape",
      `Le replay contient ${carCount} voiture(s) : le dashboard accepte de 1 à ${MAX_CARS} voitures.`,
    );
  }
  expectShape("status", status, [frameCount, carCount]);

  const simDt = Number(simDtArray.data[0]);
  if (!Number.isFinite(simDt) || simDt <= 0) {
    throw new ReplayError("bad-value", `Le pas de temps « sim_dt » est invalide : ${simDt}.`);
  }
  const declaredCars = numeric(arrays, "num_cars");
  if (declaredCars && Number(declaredCars.data[0]) !== carCount) {
    throw new ReplayError(
      "bad-value",
      `« num_cars » vaut ${declaredCars.data[0]} mais « poses » contient ${carCount} voiture(s).`,
    );
  }

  const missing: string[] = [];
  const optional = (key: string, shape: number[]): NpyArray | null => {
    const array = numeric(arrays, key);
    if (!array) {
      missing.push(key);
      return null;
    }
    expectShape(key, array, shape);
    return array;
  };
  const perCar = [frameCount, carCount];
  const velocity = optional("velocity", perCar);
  const steering = optional("steering", perCar);
  const progress = optional("progress", perCar);
  const lapCount = optional("lap_count", perCar);
  const ranks = optional("ranks", perCar);
  const maxProgress = optional("max_prog", perCar);
  const actions = optional("actions", [frameCount, carCount, 2]);
  const friction = optional("friction", [frameCount]);

  const lidar = numeric(arrays, "lidar");
  let beamCount = 0;
  if (!lidar) {
    missing.push("lidar");
  } else {
    beamCount = lidar.shape[2] ?? 0;
    expectShape("lidar", lidar, [frameCount, carCount, beamCount]);
  }

  const statusData = asInt32(status);
  const posesData = asFloat32(poses);
  const inactiveFrom = new Int32Array(carCount).fill(frameCount);
  for (let car = 0; car < carCount; car++) {
    for (let frame = 0; frame < frameCount; frame++) {
      const value = statusData[frame * carCount + car];
      if (value !== STATUS.DNF && value !== STATUS.ACTIVE && value !== STATUS.FINISHED) {
        throw new ReplayError(
          "bad-value",
          `Statut inconnu (${value}) pour la voiture ${car + 1} à la frame ${frame}.`,
        );
      }
      if (value !== STATUS.ACTIVE) {
        inactiveFrom[car] = frame;
        break;
      }
      const base = (frame * carCount + car) * 3;
      if (
        !Number.isFinite(posesData[base]) ||
        !Number.isFinite(posesData[base + 1]) ||
        !Number.isFinite(posesData[base + 2])
      ) {
        throw new ReplayError(
          "bad-value",
          `Pose non finie pour la voiture ${car + 1} à la frame ${frame}.`,
        );
      }
    }
  }

  const names = arrays.get("team_names")?.strings ?? [];
  const teamNames = Array.from({ length: carCount }, (_, car) => {
    const recorded = names[car]?.trim();
    return recorded ? recorded : `Car_${car}`;
  });
  if (!arrays.has("team_names")) missing.push("team_names");
  const mapName = arrays.get("map_name")?.strings[0]?.trim() || null;
  if (mapName === null) missing.push("map_name");

  return {
    name,
    frameCount,
    carCount,
    simDt,
    duration: (frameCount - 1) * simDt,
    mapName,
    teamNames,
    poses: posesData,
    status: statusData,
    lidar: lidar ? asFloat32(lidar) : null,
    beamCount,
    velocity: velocity ? asFloat32(velocity) : null,
    steering: steering ? asFloat32(steering) : null,
    progress: progress ? asFloat32(progress) : null,
    lapCount: lapCount ? asInt32(lapCount) : null,
    actions: actions ? asFloat32(actions) : null,
    ranks: ranks ? asInt32(ranks) : null,
    friction: friction ? asFloat32(friction) : null,
    maxProgress: maxProgress ? asFloat32(maxProgress) : null,
    missing,
    inactiveFrom,
  };
}

/** Decodes and validates the bytes of an `.npz` replay file. */
export function parseReplay(name: string, bytes: Uint8Array): Replay {
  let arrays: Map<string, NpyArray>;
  try {
    arrays = parseNpz(bytes);
  } catch (error) {
    if (error instanceof NpzError) {
      throw new ReplayError(
        "not-npz",
        "Ce fichier n'est pas une archive NPZ lisible. Choisissez un replay produit par sim_recorder.py.",
      );
    }
    throw error;
  }
  return buildReplay(name, arrays);
}
