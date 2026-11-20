/**
 * Track data of the simulator: map metadata, centerline and widths, and the lines that the
 * simulator derives from them (track edge and DNF limit). Coordinates are simulator metres,
 * x to the east and y to the north. No rendering code here.
 */

/** Distance past the track edge where the simulator declares a DNF (`_OFFTRACK_MARGIN_M`). */
export const OFFTRACK_MARGIN_M = 1.0;

export interface MapMeta {
  /** Size of one pixel of the occupancy image, in metres. */
  resolution: number;
  /** World position of the lower-left corner of the image. */
  originX: number;
  originY: number;
}

export interface Centerline {
  /** Waypoints. When `closed`, the first point is repeated at the end, as in the simulator. */
  x: Float64Array;
  y: Float64Array;
  /** Half-widths to the right and to the left of each waypoint, in metres. */
  right: Float64Array;
  left: Float64Array;
  closed: boolean;
  /** Cumulative arc length at each waypoint, in metres. */
  arc: Float64Array;
}

export class TrackDataError extends Error {
  override name = "TrackDataError";
}

export function parseMapYaml(text: string): MapMeta {
  const resolution = Number(/^\s*resolution:\s*([-+.\deE]+)/m.exec(text)?.[1]);
  const origin = /^\s*origin:\s*\[([^\]]+)\]/m.exec(text)?.[1]?.split(",").map(Number);
  const originX = origin?.[0];
  const originY = origin?.[1];
  if (
    !Number.isFinite(resolution) ||
    resolution <= 0 ||
    originX === undefined ||
    originY === undefined ||
    !Number.isFinite(originX) ||
    !Number.isFinite(originY)
  ) {
    throw new TrackDataError("map YAML without a valid resolution and origin");
  }
  return { resolution, originX, originY };
}

/** Half-width that the simulator uses when the CSV has no width columns. */
const DEFAULT_HALF_WIDTH_M = 1.05;

export function parseCenterline(csv: string): Centerline {
  const xs: number[] = [];
  const ys: number[] = [];
  const rights: number[] = [];
  const lefts: number[] = [];
  for (const line of csv.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (trimmed === "" || trimmed.startsWith("#")) continue;
    const parts = trimmed.split(/[,;]\s*|\s+/).map(Number);
    const [x, y, right, left] = parts;
    if (x === undefined || y === undefined || !Number.isFinite(x) || !Number.isFinite(y)) continue;
    xs.push(x);
    ys.push(y);
    rights.push(right !== undefined && Number.isFinite(right) ? right : DEFAULT_HALF_WIDTH_M);
    lefts.push(left !== undefined && Number.isFinite(left) ? left : DEFAULT_HALF_WIDTH_M);
  }
  if (xs.length < 3) throw new TrackDataError("centerline with fewer than 3 waypoints");

  // Same rule as `_set_waypoints`: the CSV is an open polyline, closed when its ends are near.
  const last = xs.length - 1;
  const gap = Math.hypot((xs[0] ?? 0) - (xs[last] ?? 0), (ys[0] ?? 0) - (ys[last] ?? 0));
  const firstSegment = Math.hypot((xs[1] ?? 0) - (xs[0] ?? 0), (ys[1] ?? 0) - (ys[0] ?? 0));
  const closed = gap > 0 && gap <= 2 * Math.max(firstSegment, 1e-9);
  if (closed) {
    xs.push(xs[0] ?? 0);
    ys.push(ys[0] ?? 0);
    rights.push(rights[0] ?? DEFAULT_HALF_WIDTH_M);
    lefts.push(lefts[0] ?? DEFAULT_HALF_WIDTH_M);
  }
  const arc = new Float64Array(xs.length);
  for (let i = 1; i < xs.length; i++) {
    arc[i] =
      (arc[i - 1] ?? 0) +
      Math.hypot((xs[i] ?? 0) - (xs[i - 1] ?? 0), (ys[i] ?? 0) - (ys[i - 1] ?? 0));
  }
  return {
    x: Float64Array.from(xs),
    y: Float64Array.from(ys),
    right: Float64Array.from(rights),
    left: Float64Array.from(lefts),
    closed,
    arc,
  };
}

/** Number of waypoints of the CSV file, without the point repeated to close the loop. */
function sourceCount(line: Centerline): number {
  return line.closed ? line.x.length - 1 : line.x.length;
}

/**
 * Signed lateral offset of a point from the centerline, positive on the right, and the index
 * of the nearest segment. Same computation as `_project_lateral_offset_njit` in the simulator.
 */
export function lateralOffset(
  line: Centerline,
  px: number,
  py: number,
): { lateral: number; segment: number } {
  const { x, y } = line;
  let best = 0;
  let bestT = 0;
  let bestDistance = Infinity;
  for (let i = 0; i < x.length - 1; i++) {
    const ax = x[i] ?? 0;
    const ay = y[i] ?? 0;
    const dx = (x[i + 1] ?? 0) - ax;
    const dy = (y[i + 1] ?? 0) - ay;
    const lengthSq = dx * dx + dy * dy;
    let t = lengthSq > 0 ? ((px - ax) * dx + (py - ay) * dy) / lengthSq : 0;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const ex = px - (ax + t * dx);
    const ey = py - (ay + t * dy);
    const distance = ex * ex + ey * ey;
    if (distance < bestDistance) {
      bestDistance = distance;
      best = i;
      bestT = t;
    }
  }
  const ax = x[best] ?? 0;
  const ay = y[best] ?? 0;
  const dx = (x[best + 1] ?? 0) - ax;
  const dy = (y[best + 1] ?? 0) - ay;
  const length = Math.hypot(dx, dy);
  let lateral = (px - (ax + bestT * dx)) * dy - (py - (ay + bestT * dy)) * dx;
  if (length > 0) lateral /= length;
  return { lateral, segment: best };
}

/**
 * Distance of a point past the track edge plus `margin`, in metres: positive outside. With
 * `margin = OFFTRACK_MARGIN_M`, a positive value is the off-track DNF test of the simulator.
 */
export function distancePastEdge(line: Centerline, px: number, py: number, margin: number): number {
  const { lateral, segment } = lateralOffset(line, px, py);
  // `_is_off_track` reads the widths by segment index, clamped to the rows of the CSV.
  const row = Math.min(segment, sourceCount(line) - 1);
  return Math.max(
    lateral - ((line.right[row] ?? 0) + margin),
    -lateral - ((line.left[row] ?? 0) + margin),
  );
}

/** Unit tangent of the centerline at a waypoint, averaged over the two adjacent segments. */
function tangentAt(line: Centerline, index: number): [number, number] {
  const count = line.x.length;
  let before = index - 1;
  let after = index + 1;
  if (line.closed) {
    // The last point repeats the first one: skip it when wrapping.
    if (before < 0) before = count - 2;
    if (after > count - 1) after = 1;
  } else {
    before = Math.max(before, 0);
    after = Math.min(after, count - 1);
  }
  const dx = (line.x[after] ?? 0) - (line.x[before] ?? 0);
  const dy = (line.y[after] ?? 0) - (line.y[before] ?? 0);
  const length = Math.hypot(dx, dy) || 1;
  return [dx / length, dy / length];
}

/** A kept point can be this far inside the limit: the widths change a little between waypoints. */
const FOLD_TOLERANCE_M = 0.06;

export type Side = "left" | "right";

export interface EdgeLine {
  /** x, y pairs. */
  points: Float64Array;
  /** Index of the centerline waypoint that produced each point. */
  waypoint: Int32Array;
  closed: boolean;
}

/**
 * Line at `width + margin` from the centerline on one side. An offset curve folds over itself
 * inside a turn tighter than the offset; the folded points are inside the limit for the
 * simulator, so they are removed with the same test as the simulator.
 */
export function edgeLine(line: Centerline, side: Side, margin: number): EdgeLine {
  const count = sourceCount(line);
  const points: number[] = [];
  const waypoints: number[] = [];
  const sign = side === "right" ? 1 : -1;
  for (let i = 0; i < count; i++) {
    const [tx, ty] = tangentAt(line, i);
    const width = (side === "right" ? line.right[i] : line.left[i]) ?? 0;
    const offset = width + margin;
    // Right of the direction of travel is (ty, -tx).
    const px = (line.x[i] ?? 0) + sign * ty * offset;
    const py = (line.y[i] ?? 0) - sign * tx * offset;
    if (distancePastEdge(line, px, py, margin) < -FOLD_TOLERANCE_M) continue;
    points.push(px, py);
    waypoints.push(i);
  }
  return {
    points: Float64Array.from(points),
    waypoint: Int32Array.from(waypoints),
    closed: line.closed,
  };
}

/** Unsigned curvature at each waypoint of the CSV, in rad/m, smoothed over a short window. */
export function curvature(line: Centerline): Float64Array {
  const count = sourceCount(line);
  const raw = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    const before = line.closed ? (i - 1 + count) % count : Math.max(i - 1, 0);
    const after = line.closed ? (i + 1) % count : Math.min(i + 1, count - 1);
    const ax = (line.x[i] ?? 0) - (line.x[before] ?? 0);
    const ay = (line.y[i] ?? 0) - (line.y[before] ?? 0);
    const bx = (line.x[after] ?? 0) - (line.x[i] ?? 0);
    const by = (line.y[after] ?? 0) - (line.y[i] ?? 0);
    const turn = Math.atan2(ax * by - ay * bx, ax * bx + ay * by);
    const span = (Math.hypot(ax, ay) + Math.hypot(bx, by)) / 2;
    raw[i] = span > 0 ? Math.abs(turn) / span : 0;
  }
  const smooth = new Float64Array(count);
  const radius = 3;
  for (let i = 0; i < count; i++) {
    let sum = 0;
    let weight = 0;
    for (let k = -radius; k <= radius; k++) {
      const j = line.closed ? (i + k + count) % count : i + k;
      if (j < 0 || j >= count) continue;
      sum += raw[j] ?? 0;
      weight++;
    }
    smooth[i] = sum / weight;
  }
  return smooth;
}

export interface Bounds {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export function boundsOf(points: ArrayLike<number>, stride = 2): Bounds {
  const bounds = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
  for (let i = 0; i + 1 < points.length; i += stride) {
    const x = points[i] ?? 0;
    const y = points[i + 1] ?? 0;
    if (x < bounds.minX) bounds.minX = x;
    if (x > bounds.maxX) bounds.maxX = x;
    if (y < bounds.minY) bounds.minY = y;
    if (y > bounds.maxY) bounds.maxY = y;
  }
  return bounds;
}

/**
 * Left and right track edges at each waypoint (x, y pairs), one pair per waypoint and not
 * filtered: the surface between them is the corridor that the widths describe.
 */
export function corridor(line: Centerline): { left: Float64Array; right: Float64Array } {
  const count = line.x.length;
  const left = new Float64Array(count * 2);
  const right = new Float64Array(count * 2);
  for (let i = 0; i < count; i++) {
    const [tx, ty] = tangentAt(line, i === count - 1 && line.closed ? 0 : i);
    const x = line.x[i] ?? 0;
    const y = line.y[i] ?? 0;
    const toRight = line.right[i] ?? 0;
    const toLeft = line.left[i] ?? 0;
    right[i * 2] = x + ty * toRight;
    right[i * 2 + 1] = y - tx * toRight;
    left[i * 2] = x - ty * toLeft;
    left[i * 2 + 1] = y + tx * toLeft;
  }
  return { left, right };
}

/** Position and unit tangent of the first waypoint: the start pose of the simulator. */
export function startPose(line: Centerline): { x: number; y: number; tx: number; ty: number } {
  const dx = (line.x[1] ?? 0) - (line.x[0] ?? 0);
  const dy = (line.y[1] ?? 0) - (line.y[0] ?? 0);
  const length = Math.hypot(dx, dy) || 1;
  return { x: line.x[0] ?? 0, y: line.y[0] ?? 0, tx: dx / length, ty: dy / length };
}
