/**
 * LiDAR of the selected car, in simulator coordinates: one line per recorded beam, from the
 * car to the measured range, a dot where a beam met a wall, and the swept area on the ground.
 */
import * as THREE from "three/webgpu";

import type { Pose } from "../replay/playback";
import type { Replay } from "../replay/replay";
import { beamAngles, LIDAR_MAX_RANGE_M } from "../replay/standings";
import { WALL_HEIGHT_M } from "./trackView";

/** The beams are drawn at half the wall height: they end on the face of a wall. */
const BEAM_Z = WALL_HEIGHT_M / 2;
const AREA_Z = 0.022;
const HIT_RADIUS_M = 0.035;

export class LidarView {
  readonly group = new THREE.Group();
  private readonly beams: THREE.LineSegments;
  private readonly area: THREE.Mesh;
  private readonly hits: THREE.InstancedMesh;
  private readonly beamPositions: THREE.BufferAttribute;
  private readonly areaPositions: THREE.BufferAttribute;
  private readonly cos: Float64Array;
  private readonly sin: Float64Array;
  /** End of each beam in simulator metres (x, y), at the frame before and after the playhead. */
  private readonly ends: Float64Array;
  private readonly matrix = new THREE.Matrix4();

  constructor(private readonly beamCount: number) {
    const angles = beamAngles(beamCount);
    this.cos = angles.map(Math.cos);
    this.sin = angles.map(Math.sin);
    this.ends = new Float64Array(beamCount * 2);

    this.beamPositions = new THREE.BufferAttribute(new Float32Array(beamCount * 6), 3);
    this.beamPositions.setUsage(THREE.DynamicDrawUsage);
    const beamGeometry = new THREE.BufferGeometry();
    beamGeometry.setAttribute("position", this.beamPositions);
    this.beams = new THREE.LineSegments(
      beamGeometry,
      new THREE.LineBasicMaterial({
        color: 0x7dffb0,
        transparent: true,
        opacity: 0.55,
        depthWrite: false,
      }),
    );

    // A fan: the car, then the end of each beam. The scan does not close on itself.
    this.areaPositions = new THREE.BufferAttribute(new Float32Array((beamCount + 1) * 3), 3);
    this.areaPositions.setUsage(THREE.DynamicDrawUsage);
    const areaGeometry = new THREE.BufferGeometry();
    areaGeometry.setAttribute("position", this.areaPositions);
    const fan: number[] = [];
    for (let beam = 0; beam < beamCount - 1; beam++) fan.push(0, beam + 1, beam + 2);
    areaGeometry.setIndex(fan);
    this.area = new THREE.Mesh(
      areaGeometry,
      new THREE.MeshBasicMaterial({
        color: 0x7dffb0,
        transparent: true,
        opacity: 0.1,
        depthWrite: false,
        side: THREE.DoubleSide,
      }),
    );

    this.hits = new THREE.InstancedMesh(
      new THREE.OctahedronGeometry(HIT_RADIUS_M, 1),
      new THREE.MeshBasicMaterial({ color: 0xd8ffe6 }),
      beamCount,
    );
    this.hits.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

    for (const object of [this.beams, this.area, this.hits]) {
      // The geometry changes at each frame: skip the culling test instead of new bounds.
      object.frustumCulled = false;
      this.group.add(object);
    }
    this.group.name = "lidar";
    this.group.visible = false;
  }

  /**
   * Draws the scan of `car` at the playhead. `pose` is the pose of the car on screen. At a
   * recorded frame the beams are the recorded ranges exactly. Between two frames, the end of
   * each beam moves in a straight line from its point at the frame before to its point at the
   * frame after: the ends stay on the walls while the car moves.
   */
  update(replay: Replay, frame: number, blend: number, car: number, pose: Pose): void {
    const lidar = replay.lidar;
    if (!lidar) return;
    const { beamCount, ends } = this;
    const next = Math.min(frame + 1, replay.frameCount - 1);
    const blended = blend > 0 && next !== frame && (replay.inactiveFrom[car] ?? 0) > next;
    this.writeEnds(replay, lidar, frame, car, 1);
    if (blended) this.writeEnds(replay, lidar, next, car, blend);

    const base = (frame * replay.carCount + car) * beamCount;
    this.areaPositions.setXYZ(0, pose.x, pose.y, AREA_Z);
    let hitCount = 0;
    for (let beam = 0; beam < beamCount; beam++) {
      const x = ends[beam * 2] ?? 0;
      const y = ends[beam * 2 + 1] ?? 0;
      this.beamPositions.setXYZ(beam * 2, pose.x, pose.y, BEAM_Z);
      this.beamPositions.setXYZ(beam * 2 + 1, x, y, BEAM_Z);
      this.areaPositions.setXYZ(beam + 1, x, y, AREA_Z);
      if ((lidar[base + beam] ?? 0) < LIDAR_MAX_RANGE_M - 1e-3) {
        this.hits.setMatrixAt(hitCount++, this.matrix.makeTranslation(x, y, BEAM_Z));
      }
    }
    this.hits.count = hitCount;
    this.hits.instanceMatrix.needsUpdate = true;
    this.beamPositions.needsUpdate = true;
    this.areaPositions.needsUpdate = true;
  }

  /** Mixes the beam ends of a recorded frame into `ends`, with the given weight. */
  private writeEnds(
    replay: Replay,
    lidar: Float32Array,
    frame: number,
    car: number,
    weight: number,
  ): void {
    const cell = frame * replay.carCount + car;
    const x = replay.poses[cell * 3] ?? 0;
    const y = replay.poses[cell * 3 + 1] ?? 0;
    const heading = replay.poses[cell * 3 + 2] ?? 0;
    const cosHeading = Math.cos(heading);
    const sinHeading = Math.sin(heading);
    const { ends } = this;
    for (let beam = 0; beam < this.beamCount; beam++) {
      const range = lidar[cell * this.beamCount + beam] ?? 0;
      const cos = this.cos[beam] ?? 0;
      const sin = this.sin[beam] ?? 0;
      const endX = x + range * (cosHeading * cos - sinHeading * sin);
      const endY = y + range * (sinHeading * cos + cosHeading * sin);
      ends[beam * 2] = (ends[beam * 2] ?? 0) * (1 - weight) + endX * weight;
      ends[beam * 2 + 1] = (ends[beam * 2 + 1] ?? 0) * (1 - weight) + endY * weight;
    }
  }

  dispose(): void {
    for (const object of [this.beams, this.area, this.hits]) {
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    }
  }
}
