/** Camera choice and the "Suivre" menu of the Chase camera. */
import { CAMERA_LABELS, CAMERA_MODES, type CameraMode } from "../scene/cameraDirector";
import { carColor } from "../scene/cars";
import { h, setText } from "./dom";

const SHORTCUTS: Record<CameraMode, string> = { aerial: "A", free: "F", chase: "C" };

export class CameraSwitch {
  readonly element: HTMLElement;
  private readonly buttons = new Map<CameraMode, HTMLButtonElement>();

  constructor(onPick: (mode: CameraMode) => void) {
    this.element = h("div", {
      class: "segmented cameras",
      role: "radiogroup",
      "aria-label": "Caméra",
    });
    for (const mode of CAMERA_MODES) {
      const button = h(
        "button",
        {
          type: "button",
          role: "radio",
          title: `Caméra ${CAMERA_LABELS[mode]} (${SHORTCUTS[mode]})`,
          onclick: () => onPick(mode),
        },
        CAMERA_LABELS[mode],
      );
      this.buttons.set(mode, button);
      this.element.append(button);
    }
  }

  setMode(mode: CameraMode): void {
    for (const [candidate, button] of this.buttons) {
      button.toggleAttribute("data-active", candidate === mode);
      button.setAttribute("aria-checked", String(candidate === mode));
    }
  }
}

export interface FollowHandlers {
  /** Follows the selected car again. */
  onLink(): void;
  /** Follows one car, whatever the selected car is. */
  onFollow(car: number): void;
}

/** Menu that chooses the Voiture suivie: linked to the selected car, or one fixed car. */
export class FollowMenu {
  readonly element: HTMLElement;
  private readonly button: HTMLButtonElement;
  private readonly label = h("span", {});
  private readonly menu = h("div", { class: "panel menu", role: "menu", hidden: true });
  private names: readonly string[] = [];
  private linked = true;
  private followed = 0;

  constructor(private readonly handlers: FollowHandlers) {
    this.button = h(
      "button",
      {
        class: "button follow-button",
        type: "button",
        "aria-haspopup": "menu",
        "aria-expanded": "false",
        title: "Voiture suivie par la caméra Poursuite",
        onclick: () => this.toggle(),
      },
      this.label,
    );
    this.element = h("div", { class: "follow", hidden: true }, this.button, this.menu);
    document.addEventListener("pointerdown", (event) => {
      if (!this.menu.hidden && !this.element.contains(event.target as Node)) this.close();
    });
    this.element.addEventListener("keydown", (event) => {
      if (event.key !== "Escape" || this.menu.hidden) return;
      event.stopPropagation();
      this.close();
      this.button.focus();
    });
  }

  setCars(names: readonly string[]): void {
    this.names = names;
    this.render();
  }

  setFollowed(followed: number, linked: boolean): void {
    if (followed === this.followed && linked === this.linked) return;
    this.followed = followed;
    this.linked = linked;
    this.render();
  }

  private toggle(): void {
    if (this.menu.hidden) {
      this.menu.hidden = false;
      this.button.setAttribute("aria-expanded", "true");
    } else {
      this.close();
    }
  }

  private close(): void {
    this.menu.hidden = true;
    this.button.setAttribute("aria-expanded", "false");
  }

  private render(): void {
    const name = this.names[this.followed] ?? "";
    setText(this.label, `Suivre : ${name}${this.linked ? "" : " (déliée)"}`);
    this.button.style.setProperty("--car", carColor(this.followed));
    const item = (text: string, checked: boolean, color: string | null, pick: () => void) => {
      const chip = h("span", { class: "car-chip" });
      if (color) chip.style.setProperty("--car", color);
      else chip.style.visibility = "hidden";
      return h(
        "button",
        {
          type: "button",
          role: "menuitemradio",
          "aria-checked": String(checked),
          onclick: () => {
            pick();
            this.close();
          },
        },
        chip,
        text,
      );
    };
    this.menu.replaceChildren(
      item("Voiture sélectionnée", this.linked, null, () => this.handlers.onLink()),
      h("div", { class: "menu-separator" }),
      ...this.names.map((carName, car) =>
        item(carName, !this.linked && car === this.followed, carColor(car), () =>
          this.handlers.onFollow(car),
        ),
      ),
    );
  }
}
