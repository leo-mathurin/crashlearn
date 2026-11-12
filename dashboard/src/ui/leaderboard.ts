/** Leaderboard: the cars in race order at the playhead. A click selects a car. */
import type { Replay } from "../replay/replay";
import { type Outcome, orderAt, outcomes, valuesAt } from "../replay/standings";
import { carColor } from "../scene/cars";
import { formatTime, h, setText } from "./dom";

const ROW_HEIGHT_PX = 44;

export const UNAVAILABLE = "indisponible";

interface Row {
  element: HTMLButtonElement;
  position: HTMLElement;
  detail: HTMLElement;
  place: number;
}

export class Leaderboard {
  readonly element: HTMLElement;
  private readonly title = h("h2", {}, "Classement");
  private readonly list = h("div", { class: "leaderboard-rows", role: "list" });
  private rows: Row[] = [];
  private replay: Replay | null = null;
  private results: Outcome[] = [];
  private shownFrame = -1;

  constructor(private readonly onSelect: (car: number) => void) {
    this.element = h("div", { class: "panel leaderboard", hidden: true }, this.title, this.list);
  }

  setReplay(replay: Replay): void {
    this.replay = replay;
    this.results = outcomes(replay);
    this.shownFrame = -1;
    this.rows = replay.teamNames.map((name, car) => {
      const position = h("span", { class: "row-position" });
      const detail = h("span", { class: "row-detail" });
      const chip = h("span", { class: "car-chip" });
      chip.style.setProperty("--car", carColor(car));
      const element = h(
        "button",
        {
          class: "leaderboard-row",
          type: "button",
          role: "listitem",
          title: `Sélectionner ${name} (${car + 1})`,
          onclick: () => this.onSelect(car),
        },
        position,
        chip,
        h("span", { class: "row-name" }, name),
        detail,
      );
      return { element, position, detail, place: -1 };
    });
    this.list.style.height = `${this.rows.length * ROW_HEIGHT_PX}px`;
    this.list.replaceChildren(...this.rows.map((row) => row.element));
    this.element.hidden = false;
  }

  setSelected(selected: number): void {
    this.rows.forEach((row, car) => {
      row.element.toggleAttribute("data-selected", car === selected);
      row.element.setAttribute("aria-current", String(car === selected));
    });
  }

  update(frame: number): void {
    const replay = this.replay;
    if (!replay || frame === this.shownFrame) return;
    this.shownFrame = frame;
    setText(this.title, frame === replay.frameCount - 1 ? "Classement final" : "Classement");
    orderAt(replay, frame).forEach((car, place) => {
      const row = this.rows[car];
      if (!row) return;
      if (row.place !== place) {
        row.place = place;
        row.element.style.transform = `translateY(${place * ROW_HEIGHT_PX}px)`;
        setText(row.position, String(place + 1));
      }
      const values = valuesAt(replay, frame, car);
      row.element.dataset.status = values.status;
      setText(row.detail, this.detailOf(car, values.status, values.laps, values.progress));
    });
  }

  private detailOf(
    car: number,
    status: string,
    laps: number | null,
    progress: number | null,
  ): string {
    const outcome = this.results[car];
    if (status === "finished") return `Arrivée · ${formatTime(outcome?.time ?? 0)}`;
    if (status === "dnf") return `DNF · ${formatTime(outcome?.time ?? 0)}`;
    if (laps === null || progress === null) return UNAVAILABLE;
    return `Tour ${laps + 1} · ${Math.floor(progress * 100)} %`;
  }
}
