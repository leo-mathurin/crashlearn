/** Timeline: play and pause, playhead, seek, playback speed. */
import { SPEEDS } from "../replay/playback";
import { formatTime, h, icon, ICONS, setText } from "./dom";

export interface TimelineHandlers {
  onToggle(): void;
  onSeek(time: number): void;
  onSpeed(speed: number): void;
}

export interface TimelineState {
  time: number;
  frame: number;
  playing: boolean;
  ended: boolean;
  speed: number;
}

export function formatSpeed(speed: number): string {
  return `${String(speed).replace(".", ",")}×`;
}

export class Timeline {
  readonly element: HTMLElement;
  private readonly playButton: HTMLButtonElement;
  private readonly scrubber: HTMLElement;
  private readonly fill: HTMLElement;
  private readonly thumb: HTMLElement;
  private readonly tooltip: HTMLElement;
  private readonly clock: HTMLElement;
  private readonly total: HTMLElement;
  private readonly frameLabel: HTMLElement;
  private readonly speedButtons = new Map<number, HTMLButtonElement>();
  private duration = 0;
  private frameCount = 0;
  private shownIcon = "";
  private dragging = false;

  constructor(private readonly handlers: TimelineHandlers) {
    this.playButton = h("button", {
      class: "icon-button play",
      type: "button",
      onclick: () => handlers.onToggle(),
    });
    this.clock = h("span", { class: "clock-now" }, "0:00.0");
    this.total = h("span", { class: "clock-total" }, "0:00.0");
    this.fill = h("div", { class: "scrub-fill" });
    this.thumb = h("div", { class: "scrub-thumb" });
    this.tooltip = h("div", { class: "scrub-tooltip" });
    this.scrubber = h(
      "div",
      {
        class: "scrubber",
        role: "slider",
        tabindex: 0,
        "aria-label": "Position dans la course",
        "aria-valuemin": 0,
      },
      h("div", { class: "scrub-track" }, this.fill),
      this.thumb,
      this.tooltip,
    );
    this.frameLabel = h("span", { class: "frame-label", title: "Frame enregistrée" });
    const speeds = h("div", { class: "segmented speeds", role: "group", "aria-label": "Vitesse" });
    for (const speed of SPEEDS) {
      const button = h(
        "button",
        { type: "button", onclick: () => handlers.onSpeed(speed) },
        formatSpeed(speed),
      );
      this.speedButtons.set(speed, button);
      speeds.append(button);
    }
    this.element = h(
      "div",
      { class: "panel timeline" },
      this.playButton,
      h("div", { class: "clock" }, this.clock, h("span", { class: "clock-sep" }, "/"), this.total),
      this.scrubber,
      this.frameLabel,
      speeds,
    );
    this.bindScrubber();
  }

  setReplay(duration: number, frameCount: number): void {
    this.duration = duration;
    this.frameCount = frameCount;
    setText(this.total, formatTime(duration));
    this.scrubber.setAttribute("aria-valuemax", duration.toFixed(2));
  }

  update(state: TimelineState): void {
    const ratio = this.duration > 0 ? state.time / this.duration : 0;
    this.fill.style.transform = `scaleX(${ratio})`;
    this.thumb.style.left = `${ratio * 100}%`;
    setText(this.clock, formatTime(state.time));
    setText(this.frameLabel, `${state.frame + 1} / ${this.frameCount}`);
    this.scrubber.setAttribute("aria-valuenow", state.time.toFixed(2));
    this.scrubber.setAttribute("aria-valuetext", formatTime(state.time));

    const wanted = state.playing ? "pause" : state.ended ? "restart" : "play";
    if (wanted !== this.shownIcon) {
      this.shownIcon = wanted;
      this.playButton.replaceChildren(icon(ICONS[wanted], 18));
      const label = { pause: "Pause", restart: "Revoir depuis le début", play: "Lecture" }[wanted];
      this.playButton.setAttribute("aria-label", label);
      this.playButton.title = `${label} (Espace)`;
    }
    for (const [speed, button] of this.speedButtons) {
      button.toggleAttribute("data-active", speed === state.speed);
    }
  }

  private timeAt(clientX: number): number {
    const box = this.scrubber.getBoundingClientRect();
    const ratio = Math.min(Math.max((clientX - box.left) / box.width, 0), 1);
    return ratio * this.duration;
  }

  private bindScrubber(): void {
    const { scrubber, tooltip } = this;
    const showTooltip = (clientX: number): void => {
      const time = this.timeAt(clientX);
      tooltip.textContent = formatTime(time);
      tooltip.style.left = `${this.duration > 0 ? (time / this.duration) * 100 : 0}%`;
    };
    scrubber.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      this.dragging = true;
      scrubber.setPointerCapture(event.pointerId);
      scrubber.toggleAttribute("data-dragging", true);
      this.handlers.onSeek(this.timeAt(event.clientX));
      showTooltip(event.clientX);
    });
    scrubber.addEventListener("pointermove", (event) => {
      showTooltip(event.clientX);
      if (this.dragging) this.handlers.onSeek(this.timeAt(event.clientX));
    });
    const release = (): void => {
      this.dragging = false;
      scrubber.toggleAttribute("data-dragging", false);
    };
    scrubber.addEventListener("pointerup", release);
    scrubber.addEventListener("pointercancel", release);
  }
}
