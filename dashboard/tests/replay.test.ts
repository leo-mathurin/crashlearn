import { describe, expect, test } from "bun:test";
import { zipSync } from "fflate";

import { type NpyArray, NpzError, parseNpy, parseNpz } from "../src/replay/npz";
import { frameAt, lastActivePose, Playback, type Pose, samplePose } from "../src/replay/playback";
import { buildReplay, parseReplay, ReplayError, STATUS } from "../src/replay/replay";

const fixture = new Uint8Array(
  await Bun.file(new URL("./fixtures/two-cars-40-frames.npz", import.meta.url)).arrayBuffer(),
);

/** Encodes a float32 or int32 array as a version 1 `.npy` file. */
function npy(descr: "<f4" | "<i4", shape: number[], values: number[]): Uint8Array {
  const dims = shape.length === 1 ? `${shape[0]},` : shape.join(", ");
  let header = `{'descr': '${descr}', 'fortran_order': False, 'shape': (${dims}), }`;
  header += " ".repeat((64 - ((10 + header.length + 1) % 64)) % 64) + "\n";
  const data = descr === "<f4" ? new Float32Array(values) : new Int32Array(values);
  const bytes = new Uint8Array(10 + header.length + data.byteLength);
  bytes.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, header.length & 255, header.length >> 8]);
  bytes.set(new TextEncoder().encode(header), 10);
  bytes.set(new Uint8Array(data.buffer), 10 + header.length);
  return bytes;
}

function array(descr: "<f4" | "<i4", shape: number[], values: number[]): NpyArray {
  return parseNpy(npy(descr, shape, values));
}

/** A two-frame, one-car replay with only the mandatory arrays, as an old recorder could write. */
function minimalArrays(): Map<string, NpyArray> {
  return new Map([
    ["poses", array("<f4", [2, 1, 3], [0, 0, 0, 1, 0, 0])],
    ["status", array("<i4", [2, 1], [1, 1])],
    ["sim_dt", array("<f4", [], [0.05])],
  ]);
}

describe("npz decoder", () => {
  test("reads the arrays of a recorded replay", () => {
    const arrays = parseNpz(fixture);
    expect(arrays.get("poses")?.shape).toEqual([40, 2, 3]);
    expect(arrays.get("lidar")?.shape).toEqual([40, 2, 100]);
    expect(arrays.get("lidar")?.data).toBeInstanceOf(Float32Array);
    expect(arrays.get("status")?.data).toBeInstanceOf(Int32Array);
    expect(arrays.get("map_name")?.strings).toEqual(["Spielberg"]);
    expect(arrays.get("team_names")?.strings).toEqual(["1-driver1", "2-driver2"]);
    expect(arrays.get("sim_dt")?.data[0]).toBeCloseTo(0.05);
  });

  test("refuses a file that is not a ZIP archive", () => {
    expect(() => parseNpz(new TextEncoder().encode("not a replay"))).toThrow(NpzError);
  });

  test("refuses an archive member that is not a .npy array", () => {
    const archive = zipSync({ "poses.npy": new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]) });
    expect(() => parseNpz(archive)).toThrow("bad magic number");
  });

  test("refuses an array shorter than its header says", () => {
    const short = npy("<f4", [4], [1, 2, 3, 4]).slice(0, -8);
    expect(() => parseNpy(short)).toThrow("truncated array");
  });
});

describe("replay validation", () => {
  test("accepts a recorded replay and keeps its values", () => {
    const replay = parseReplay("two-cars.npz", fixture);
    expect(replay.frameCount).toBe(40);
    expect(replay.carCount).toBe(2);
    expect(replay.simDt).toBeCloseTo(0.05);
    expect(replay.duration).toBeCloseTo(39 * 0.05);
    expect(replay.mapName).toBe("Spielberg");
    expect(replay.teamNames).toEqual(["1-driver1", "2-driver2"]);
    expect(replay.beamCount).toBe(100);
    expect(replay.missing).toEqual([]);
    expect(Array.from(replay.inactiveFrom)).toEqual([40, 40]);
  });

  test("lists the optional arrays that an old file does not have", () => {
    const replay = buildReplay("old.npz", minimalArrays());
    expect(replay.lidar).toBeNull();
    expect(replay.ranks).toBeNull();
    expect(replay.mapName).toBeNull();
    expect(replay.teamNames).toEqual(["Car_0"]);
    expect(replay.missing).toContain("lidar");
    expect(replay.missing).toContain("ranks");
    expect(replay.missing).toContain("map_name");
  });

  test("finds the frame where each car stops being active", () => {
    const arrays = minimalArrays();
    arrays.set(
      "poses",
      array(
        "<f4",
        [3, 2, 3],
        Array.from({ length: 18 }, () => 0),
      ),
    );
    arrays.set("status", array("<i4", [3, 2], [1, 1, 1, STATUS.DNF, STATUS.FINISHED, STATUS.DNF]));
    expect(Array.from(buildReplay("r.npz", arrays).inactiveFrom)).toEqual([2, 1]);
  });

  test.each([
    ["not an archive", () => parseReplay("x.npz", new Uint8Array([1, 2, 3])), "not-npz"],
    [
      "no poses",
      () => {
        const arrays = minimalArrays();
        arrays.delete("poses");
        return buildReplay("x.npz", arrays);
      },
      "missing-array",
    ],
    [
      "status of another shape",
      () => {
        const arrays = minimalArrays();
        arrays.set("status", array("<i4", [3, 1], [1, 1, 1]));
        return buildReplay("x.npz", arrays);
      },
      "bad-shape",
    ],
    [
      "five cars",
      () => {
        const arrays = minimalArrays();
        arrays.set(
          "poses",
          array(
            "<f4",
            [1, 5, 3],
            Array.from({ length: 15 }, () => 0),
          ),
        );
        arrays.set("status", array("<i4", [1, 5], [1, 1, 1, 1, 1]));
        return buildReplay("x.npz", arrays);
      },
      "bad-shape",
    ],
    [
      "zero time step",
      () => {
        const arrays = minimalArrays();
        arrays.set("sim_dt", array("<f4", [], [0]));
        return buildReplay("x.npz", arrays);
      },
      "bad-value",
    ],
    [
      "unknown status",
      () => {
        const arrays = minimalArrays();
        arrays.set("status", array("<i4", [2, 1], [1, 7]));
        return buildReplay("x.npz", arrays);
      },
      "bad-value",
    ],
    [
      "pose that is not finite",
      () => {
        const arrays = minimalArrays();
        arrays.set("poses", array("<f4", [2, 1, 3], [0, 0, 0, Number.NaN, 0, 0]));
        return buildReplay("x.npz", arrays);
      },
      "bad-value",
    ],
  ])("refuses a replay with %s", (_label, build, code) => {
    let caught: unknown;
    try {
      build();
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(ReplayError);
    expect((caught as ReplayError).code).toBe(code as ReplayError["code"]);
  });
});

describe("playback", () => {
  // One car: active for 3 frames along x with a heading that crosses +pi/-pi, then finished
  // and moved away by the simulator.
  const arrays = minimalArrays();
  arrays.set(
    "poses",
    array("<f4", [4, 1, 3], [0, 0, 3.1, 1, 0, -3.1, 2, 0, -3.1, 10000, 10000, 0]),
  );
  arrays.set("status", array("<i4", [4, 1], [1, 1, 1, STATUS.FINISHED]));
  const replay = buildReplay("r.npz", arrays);
  const pose: Pose = { x: 0, y: 0, heading: 0 };

  test("interpolates the pose between two active frames", () => {
    expect(samplePose(replay, 0.025, 0, pose)).toBe(true);
    expect(pose.x).toBeCloseTo(0.5);
    // The heading takes the short way through pi, not the long way through 0.
    expect(Math.abs(pose.heading)).toBeGreaterThan(3.1);
  });

  test("never interpolates towards the place where the simulator parks an inactive car", () => {
    expect(samplePose(replay, 0.125, 0, pose)).toBe(true);
    expect(pose.x).toBeCloseTo(2);
    expect(samplePose(replay, 0.15, 0, pose)).toBe(false);
  });

  test("gives the last pose of a car while it was active", () => {
    expect(lastActivePose(replay, 0, pose)).toBe(true);
    expect(pose.x).toBeCloseTo(2);
  });

  test("maps a time to the last frame at or before it", () => {
    expect(frameAt(replay, 0)).toBe(0);
    expect(frameAt(replay, 0.0999)).toBe(1);
    expect(frameAt(replay, 0.1)).toBe(2);
    expect(frameAt(replay, 99)).toBe(3);
    expect(frameAt(replay, -1)).toBe(0);
  });

  test("stops at the end and restarts from the beginning", () => {
    const playback = new Playback(replay);
    playback.toggle();
    playback.speed = 2;
    playback.advance(0.05);
    expect(playback.time).toBeCloseTo(0.1);
    playback.advance(10);
    expect(playback.time).toBeCloseTo(replay.duration);
    expect(playback.playing).toBe(false);
    expect(playback.ended).toBe(true);
    playback.toggle();
    expect(playback.time).toBe(0);
    expect(playback.playing).toBe(true);
  });

  test("steps frame by frame and stays inside the replay", () => {
    const playback = new Playback(replay);
    playback.seek(0.07);
    playback.step(1);
    expect(playback.frame).toBe(2);
    playback.step(-10);
    expect(playback.frame).toBe(0);
    playback.step(100);
    expect(playback.frame).toBe(3);
    expect(playback.playing).toBe(false);
  });
});
