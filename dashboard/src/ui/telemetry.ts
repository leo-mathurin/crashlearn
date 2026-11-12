/** Telemetry of the selected car at the playhead: speed, steering, lap, and a speed chart. */
import { frameAt } from "../replay/playback";
import type { Replay } from "../replay/replay";
import { type CarValues, valuesAt } from "../replay/standings";
import { carColor } from "../scene/cars";
import { h, setText } from "./dom";
import { UNAVAILABLE } from "./leaderboard";

/** Bounds of the simulator commands: `apply_action` clamps to these values. */
const SPEED_MAX_MPS = 10;
const STEERING_MAX_RAD = 0.4189;
/** The chart shows this many seconds before and after the playhead. */
const CHART_WINDOW_S = 6;

const STATUS_LABEL = { active: "En course", finished: "Arrivée", dnf: "DNF" } as const;

function decimal(value: number, digits: number): string {
  return value.toFixed(digits).replace(".", ",").replace("-", "−");
}

function degrees(radians: number): string {
  return `${decimal((radians * 180) / Math.PI, 1)}°`;
}

interface Gauge {
  element: HTMLElement;
  value: HTMLElement;
  target: HTMLElement;
  fill: HTMLElement;
  tick: HTMLElement;
}

function gauge(label: string, unit: string, centered: boolean): Gauge {
  const value = h("span", { class: "gauge-value" });
  const target = h("span", { class: "gauge-target" });
  const fill = h("div", { class: "gauge-fill" });
  const tick = h("div", { class: "gauge-tick", title: "Consigne du pilote" });
  const element = h(
    "div",
    { class: "gauge" },
    h(
      "div",
      { class: "gauge-head" },
      h("span", { class: "gauge-label" }, label),
      h(
        "span",
        { class: "gauge-reading" },
        value,
        unit ? h("span", { class: "muted" }, unit) : null,
      ),
    ),
    h("div", { class: centered ? "gauge-track centered" : "gauge-track" }, fill, tick),
    target,
  );
  return { element, value, target, fill, tick };
}

export class Telemetry {
  readonly element: HTMLElement;
  private readonly chip = h("span", { class: "car-chip" });
  private readonly name = h("span", { class: "telemetry-name" });
  private readonly status = h("span", { class: "status-pill" });
  private readonly speed = gauge("Vitesse", "m/s", false);
  private readonly steering = gauge("Braquage", "", true);
  private readonly lap = h("span", {});
  private readonly friction = h("span", {});
  private readonly progress = h("div", { class: "gauge-fill" });
  private readonly canvas = h("canvas", { class: "speed-chart", "aria-hidden": "true" });
  private readonly chartNote = h("span", { class: "muted chart-note" });
  private replay: Replay | null = null;
  private car = 0;
  private shownKey = "";

  constructor() {
    this.steering.element.classList.add("gauge-steering");
    this.element = h(
      "div",
      { class: "panel telemetry", hidden: true },
      h("header", {}, this.chip, this.name, this.status),
      this.speed.element,
      this.steering.element,
      h(
        "div",
        { class: "gauge gauge-lap" },
        h(
          "div",
          { class: "gauge-head" },
          h("span", { class: "gauge-label" }, "Tour"),
          h("span", { class: "gauge-reading small" }, this.lap),
        ),
        h("div", { class: "gauge-track" }, this.progress),
      ),
      h(
        "div",
        { class: "telemetry-line" },
        h("span", { class: "gauge-label" }, "Adhérence"),
        this.friction,
      ),
      h(
        "div",
        { class: "chart" },
        h(
          "div",
          { class: "chart-legend" },
          h("span", { class: "legend-real" }, "Vitesse réelle"),
          h("span", { class: "legend-target" }, "Vitesse cible"),
          this.chartNote,
        ),
        this.canvas,
      ),
    );
  }

  setReplay(replay: Replay): void {
    this.replay = replay;
    this.shownKey = "";
    this.element.hidden = false;
  }

  setCar(car: number): void {
    this.car = car;
    this.shownKey = "";
    this.element.style.setProperty("--car", carColor(car));
    setText(this.name, this.replay?.teamNames[car] ?? "");
  }

  /** The values on screen, for the tests. */
  values(time: number): CarValues | null {
    return this.replay ? valuesAt(this.replay, frameAt(this.replay, time), this.car) : null;
  }

  update(time: number): void {
    const replay = this.replay;
    if (!replay) return;
    const frame = frameAt(replay, time);
    const key = `${this.car}:${frame}`;
    if (key === this.shownKey) return;
    this.shownKey = key;

    const values = valuesAt(replay, frame, this.car);
    setText(this.status, STATUS_LABEL[values.status]);
    this.status.dataset.status = values.status;
    // After a DNF or the finish, the simulator freezes the car: its values are no longer a
    // measurement of the race.
    const live = values.status === "active";
    this.element.toggleAttribute("data-inactive", !live);

    this.showGauge(this.speed, live ? values.speed : null, values.targetSpeed, (value) => {
      const share = Math.min(Math.max(value / SPEED_MAX_MPS, 0), 1);
      return { text: decimal(value, 2), from: 0, to: share, at: share };
    });
    this.showGauge(this.steering, live ? values.steering : null, values.targetSteering, (value) => {
      // Positive steering turns left: the bar grows to the left of the centre.
      const share = Math.min(Math.max(value / STEERING_MAX_RAD, -1), 1) / 2;
      return {
        text: degrees(value),
        from: 0.5 - Math.max(share, 0),
        to: 0.5 - Math.min(share, 0),
        at: 0.5 - share,
      };
    });

    if (values.laps === null || values.progress === null) {
      setText(this.lap, UNAVAILABLE);
      this.progress.style.transform = "scaleX(0)";
    } else {
      setText(this.lap, `${values.laps + 1} · ${Math.floor(values.progress * 100)} %`);
      this.progress.style.transform = `scaleX(${live ? values.progress : 0})`;
    }
    setText(this.friction, values.friction === null ? UNAVAILABLE : decimal(values.friction, 2));
    this.drawChart(replay, frame);
  }

  private showGauge(
    target: Gauge,
    value: number | null,
    wanted: number | null,
    scale: (value: number) => { text: string; from: number; to: number; at: number },
  ): void {
    if (value === null) {
      setText(target.value, "–");
      target.fill.style.transform = "scaleX(0)";
    } else {
      const shown = scale(value);
      setText(target.value, shown.text);
      target.fill.style.left = `${shown.from * 100}%`;
      target.fill.style.transform = `scaleX(${shown.to - shown.from})`;
    }
    if (wanted === null || value === null) {
      setText(target.target, value === null ? "" : `Consigne ${UNAVAILABLE}`);
      target.tick.hidden = true;
    } else {
      const shown = scale(wanted);
      setText(target.target, `Consigne ${shown.text}`);
      target.tick.hidden = false;
      target.tick.style.left = `${shown.at * 100}%`;
    }
  }

  /** Measured and target speed around the playhead. */
  private drawChart(replay: Replay, frame: number): void {
    const { canvas } = this;
    const context = canvas.getContext("2d");
    const { velocity, actions } = replay;
    if (!context) return;
    const ratio = Math.min(window.devicePixelRatio, 2);
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (canvas.width !== Math.round(width * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    setText(this.chartNote, velocity ? "" : UNAVAILABLE);
    if (!velocity) return;

    const span = Math.round(CHART_WINDOW_S / replay.simDt);
    const end = (replay.inactiveFrom[this.car] ?? replay.frameCount) - 1;
    const xOf = (at: number): number => ((at - (frame - span)) / (span * 2)) * width;
    const yOf = (speed: number): number => height - 3 - (speed / SPEED_MAX_MPS) * (height - 6);
    const trace = (read: (at: number) => number): void => {
      context.beginPath();
      let started = false;
      for (let at = Math.max(frame - span, 0); at <= Math.min(frame + span, end); at++) {
        const x = xOf(at);
        const y = yOf(read(at));
        if (started) context.lineTo(x, y);
        else context.moveTo(x, y);
        started = true;
      }
      context.stroke();
    };

    context.lineWidth = 1;
    context.strokeStyle = "rgb(255 255 255 / 0.08)";
    for (const speed of [0, 5, 10]) {
      context.beginPath();
      context.moveTo(0, yOf(speed));
      context.lineTo(width, yOf(speed));
      context.stroke();
    }
    if (actions) {
      context.strokeStyle = "rgb(255 255 255 / 0.5)";
      context.setLineDash([3, 3]);
      trace((at) => actions[(at * replay.carCount + this.car) * 2] ?? 0);
      context.setLineDash([]);
    }
    context.strokeStyle = carColor(this.car);
    context.lineWidth = 1.75;
    trace((at) => velocity[at * replay.carCount + this.car] ?? 0);
    // Playhead.
    context.strokeStyle = "rgb(255 255 255 / 0.7)";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(width / 2, 0);
    context.lineTo(width / 2, height);
    context.stroke();
  }
}
