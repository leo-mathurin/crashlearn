import { describe, expect, test } from "bun:test";

import { parseNpy } from "../src/replay/npz";
import { buildReplay, parseReplay, STATUS } from "../src/replay/replay";
import { beamAngles, orderAt, outcomes, statusAt, valuesAt } from "../src/replay/standings";
import expected from "./fixtures/two-cars-40-frames.expected.json";

const replay = parseReplay(
  "two-cars.npz",
  new Uint8Array(
    await Bun.file(new URL("./fixtures/two-cars-40-frames.npz", import.meta.url)).arrayBuffer(),
  ),
);

function npy(descr: "<f4" | "<i4", shape: number[], values: number[]) {
  const dims = shape.length === 1 ? `${shape[0]},` : shape.join(", ");
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${dims}), }`;
  header += `${" ".repeat((64 - ((10 + header.length + 1) % 64)) % 64)}\n`;
  const data = descr === "<f4" ? new Float32Array(values) : new Int32Array(values);
  const bytes = new Uint8Array(10 + header.length + data.byteLength);
  bytes.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, header.length & 255, header.length >> 8]);
  bytes.set(new TextEncoder().encode(header), 10);
  bytes.set(new Uint8Array(data.buffer), 10 + header.length);
  return parseNpy(bytes);
}

/** Three cars over three frames: car 0 finishes at frame 2, car 2 has a DNF at frame 1. */
function race(withRanks: boolean) {
  const arrays = new Map([
    [
      "poses",
      npy(
        "<f4",
        [3, 3, 3],
        Array.from({ length: 27 }, () => 0),
      ),
    ],
    ["status", npy("<i4", [3, 3], [1, 1, 1, 1, 1, STATUS.DNF, STATUS.FINISHED, 1, STATUS.DNF])],
    ["sim_dt", npy("<f4", [], [0.05])],
    ["lap_count", npy("<i4", [3, 3], [0, 0, 0, 0, 1, 0, 1, 1, 0])],
    ["progress", npy("<f4", [3, 3], [0.2, 0.5, 0.4, 0.9, 0.1, 0.45, 0, 0.3, 0.45])],
  ]);
  if (withRanks) arrays.set("ranks", npy("<i4", [3, 3], [3, 1, 2, 2, 1, 3, 1, 2, 3]));
  return buildReplay("race.npz", arrays);
}

describe("values of a recorded replay, against NumPy", () => {
  for (const [frameText, frameExpected] of Object.entries(expected.frames)) {
    const frame = Number(frameText);
    test(`frame ${frame}: status, lap, speed, steering and commands of each car`, () => {
      frameExpected.cars.forEach((car, index) => {
        const values = valuesAt(replay, frame, index);
        expect(values.status).toBe(car.status === STATUS.ACTIVE ? "active" : "dnf");
        expect(values.laps).toBe(car.laps);
        expect(values.progress).toBeCloseTo(car.progress, 6);
        expect(values.speed).toBeCloseTo(car.speed, 6);
        expect(values.targetSpeed).toBeCloseTo(car.targetSpeed, 6);
        expect(values.steering).toBeCloseTo(car.steering, 6);
        expect(values.targetSteering).toBeCloseTo(car.targetSteering, 6);
        expect(values.friction).toBeCloseTo(frameExpected.friction, 6);
        expect(orderAt(replay, frame).indexOf(index) + 1).toBe(car.rank);
      });
    });

    test(`frame ${frame}: LiDAR beams end where the simulator geometry puts them`, () => {
      const angles = beamAngles(replay.beamCount);
      frameExpected.cars.forEach((car, index) => {
        const [x = 0, y = 0, heading = 0] = car.pose;
        for (const [beam = 0, endX = 0, endY = 0] of car.beamEnds) {
          const cell = frame * replay.carCount + index;
          const range = replay.lidar?.[cell * replay.beamCount + beam] ?? 0;
          const angle = heading + (angles[beam] ?? 0);
          expect(x + range * Math.cos(angle)).toBeCloseTo(endX, 4);
          expect(y + range * Math.sin(angle)).toBeCloseTo(endY, 4);
        }
      });
    });
  }
});

describe("LiDAR beam angles", () => {
  test("cover a full turn less one step, centred on the heading", () => {
    const angles = beamAngles(100);
    const step = (2 * Math.PI) / 100;
    expect(angles[0]).toBeCloseTo(-Math.PI + step / 2);
    expect(angles[99]).toBeCloseTo(Math.PI - step / 2);
    expect((angles[1] ?? 0) - (angles[0] ?? 0)).toBeCloseTo(step);
  });
});

describe("race order", () => {
  test("follows the ranks of the simulator when the file has them", () => {
    const ranked = race(true);
    expect(orderAt(ranked, 0)).toEqual([1, 2, 0]);
    expect(orderAt(ranked, 1)).toEqual([1, 0, 2]);
    expect(orderAt(ranked, 2)).toEqual([0, 1, 2]);
  });

  test("without ranks: finished cars, then active cars by distance, then DNF", () => {
    const old = race(false);
    expect(orderAt(old, 0)).toEqual([1, 2, 0]);
    expect(orderAt(old, 1)).toEqual([1, 0, 2]);
    expect(orderAt(old, 2)).toEqual([0, 1, 2]);
  });
});

describe("outcome of each car", () => {
  const results = outcomes(race(true));

  test("gives the end of a finished car as the step count times the time step", () => {
    expect(results[0]?.status).toBe("finished");
    expect(results[0]?.frame).toBe(2);
    expect(results[0]?.time).toBeCloseTo(3 * 0.05);
  });

  test("keeps a car that still races at the last frame as active", () => {
    expect(results[1]?.status).toBe("active");
    expect(results[1]?.frame).toBe(-1);
  });

  test("marks a DNF at the first frame where the simulator reports it", () => {
    expect(results[2]?.status).toBe("dnf");
    expect(results[2]?.frame).toBe(1);
    expect(statusAt(race(true), 0, 2)).toBe("active");
  });
});

describe("values that an old file does not have", () => {
  test("are null, not zero", () => {
    const arrays = new Map([
      ["poses", npy("<f4", [1, 1, 3], [0, 0, 0])],
      ["status", npy("<i4", [1, 1], [1])],
      ["sim_dt", npy("<f4", [], [0.05])],
    ]);
    expect(valuesAt(buildReplay("old.npz", arrays), 0, 0)).toEqual({
      status: "active",
      laps: null,
      progress: null,
      speed: null,
      targetSpeed: null,
      steering: null,
      targetSteering: null,
      friction: null,
    });
  });
});
