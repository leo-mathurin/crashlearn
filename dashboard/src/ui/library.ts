/** Replay picker: local replays served by the dashboard server, file chooser, error message. */
import { h, icon, ICONS } from "./dom";

export interface ReplayEntry {
  name: string;
  size: number;
  modified: number;
}

export interface LibraryHandlers {
  onPickServerReplay(name: string): void;
  onPickFile(file: File): void;
  onClose(): void;
}

function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} ko`;
  return `${(bytes / 1024 / 1024).toFixed(1).replace(".", ",")} Mo`;
}

const dateFormat = new Intl.DateTimeFormat("fr-FR", { dateStyle: "medium", timeStyle: "short" });

export class Library {
  readonly element: HTMLElement;
  private readonly list = h("ul", { class: "replay-list" });
  private readonly error = h("div", { class: "notice error", role: "alert", hidden: true });
  private readonly closeButton: HTMLButtonElement;
  private readonly input: HTMLInputElement;

  constructor(private readonly handlers: LibraryHandlers) {
    this.input = h("input", {
      type: "file",
      accept: ".npz",
      hidden: true,
      onchange: () => {
        const file = this.input.files?.[0];
        if (file) handlers.onPickFile(file);
        this.input.value = "";
      },
    });
    this.closeButton = h(
      "button",
      {
        class: "icon-button",
        type: "button",
        "aria-label": "Fermer",
        hidden: true,
        onclick: () => handlers.onClose(),
      },
      icon(ICONS.close, 14),
    );
    const card = h(
      "div",
      { class: "panel library-card", role: "dialog", "aria-label": "Ouvrir un replay" },
      h(
        "header",
        {},
        h(
          "div",
          {},
          h("h1", {}, "Ouvrir un replay"),
          h("p", { class: "muted" }, "Une course enregistrée par sim_recorder.py, au format .npz."),
        ),
        this.closeButton,
      ),
      this.error,
      h(
        "button",
        { class: "drop-zone", type: "button", onclick: () => this.input.click() },
        icon(ICONS.folder, 18),
        h("span", {}, "Choisir un fichier .npz"),
        h("span", { class: "muted" }, "ou le déposer dans la fenêtre"),
      ),
      this.input,
      h("h2", {}, "Replays locaux"),
      this.list,
    );
    this.element = h("div", { class: "library", hidden: true }, card);
    // A click outside the card closes the picker, when a replay is already open.
    this.element.addEventListener("pointerdown", (event) => {
      if (event.target === this.element && !this.closeButton.hidden) handlers.onClose();
    });
  }

  get open(): boolean {
    return !this.element.hidden;
  }

  /** Shows the picker. `closable` is true when a replay is already open behind it. */
  show(closable: boolean): void {
    this.closeButton.hidden = !closable;
    this.element.hidden = false;
    void this.refresh();
  }

  hide(): void {
    this.element.hidden = true;
    this.setError(null);
  }

  setError(message: string | null, detail?: string): void {
    this.error.hidden = message === null;
    this.error.replaceChildren(h("strong", {}, message ?? ""), h("span", {}, detail ?? ""));
  }

  async refresh(): Promise<void> {
    let entries: ReplayEntry[] = [];
    let directory = "replays/";
    try {
      const response = await fetch("/api/replays");
      const body: { directory: string; replays: ReplayEntry[] } = await response.json();
      entries = body.replays;
      directory = body.directory;
    } catch {
      // The page also works without the list: a file can still be chosen or dropped.
    }
    if (entries.length === 0) {
      this.list.replaceChildren(
        h(
          "li",
          { class: "empty" },
          h("span", {}, `Aucun replay dans ${directory}. Pour en enregistrer :`),
          h("code", {}, "uv run python scripts/record_replay.py --demo"),
        ),
      );
      return;
    }
    this.list.replaceChildren(
      ...entries.map((entry) =>
        h(
          "li",
          {},
          h(
            "button",
            { type: "button", onclick: () => this.handlers.onPickServerReplay(entry.name) },
            h("span", { class: "replay-name" }, entry.name.replace(/\.npz$/, "")),
            h(
              "span",
              { class: "muted" },
              `${formatSize(entry.size)} · ${dateFormat.format(entry.modified)}`,
            ),
          ),
        ),
      ),
    );
  }
}
