/** Frame time meter: a rolling window of the time between frames and of the work per frame. */
import { h, setText } from "./dom";

const WINDOW = 240;

export interface PerfSummary {
  /** Frames per second over the window. */
  fps: number;
  /** Mean and 95th percentile of the time between two frames, in milliseconds. */
  frameMs: number;
  frameP95Ms: number;
  /** Mean main-thread time to update and submit one frame, in milliseconds. */
  workMs: number;
  samples: number;
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export class PerfMeter {
  readonly element = h("div", { class: "panel perf", hidden: true });
  private readonly intervals: number[] = [];
  private readonly work: number[] = [];
  private lastShown = 0;

  /** Adds one frame: the time since the previous frame and the time spent in the frame loop. */
  sample(intervalMs: number, workMs: number, now: number, detail: string): void {
    this.intervals.push(intervalMs);
    this.work.push(workMs);
    if (this.intervals.length > WINDOW) {
      this.intervals.shift();
      this.work.shift();
    }
    if (this.element.hidden || now - this.lastShown < 500) return;
    this.lastShown = now;
    const summary = this.summary();
    setText(
      this.element,
      `${summary.fps.toFixed(0)} fps · ${summary.frameMs.toFixed(1)} ms (p95 ${summary.frameP95Ms.toFixed(1)}) · CPU ${summary.workMs.toFixed(2)} ms · ${detail}`,
    );
  }

  reset(): void {
    this.intervals.length = 0;
    this.work.length = 0;
  }

  summary(): PerfSummary {
    const count = this.intervals.length;
    if (count === 0) return { fps: 0, frameMs: 0, frameP95Ms: 0, workMs: 0, samples: 0 };
    const sorted = this.intervals.toSorted((a, b) => a - b);
    const frameMs = mean(this.intervals);
    return {
      fps: 1000 / frameMs,
      frameMs,
      frameP95Ms: sorted[Math.min(count - 1, Math.floor(count * 0.95))] ?? frameMs,
      workMs: mean(this.work),
      samples: count,
    };
  }

  toggle(): void {
    this.element.hidden = !this.element.hidden;
  }
}
