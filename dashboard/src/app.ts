/** The dashboard: opens a replay, builds its scene and runs the frame loop. */
import * as THREE from "three/webgpu";

import { Playback, type Pose, samplePose, SPEEDS } from "./replay/playback";
import { parseReplay, type Replay, ReplayError } from "./replay/replay";
import { AerialCamera } from "./scene/aerialCamera";
import { CAR_LENGTH_M, CarView } from "./scene/cars";
import { Stage } from "./scene/stage";
import { buildFallbackGround, buildTrackView, type TrackView } from "./scene/trackView";
import { loadTrackAssets, MapNotFoundError } from "./track/assets";
import { type Bounds, boundsOf } from "./track/track";
import { h, icon, ICONS, setText } from "./ui/dom";
import { Library } from "./ui/library";
import { CarMarkers } from "./ui/markers";
import { PerfMeter, type PerfSummary } from "./ui/perf";
import { Timeline } from "./ui/timeline";

interface Session {
  replay: Replay;
  playback: Playback;
  track: TrackView;
  cars: CarView[];
  /** False when the map of the replay was not found: the circuit is not drawn. */
  mapAvailable: boolean;
}

/** Height of the HUD bars over the scene: the circuit is framed between them. */
const HUD_TOP_PX = 70;
const HUD_BOTTOM_PX = 84;

/** State that the browser tests read through `window.dashboard`. */
export interface DashboardState {
  phase: "library" | "loading" | "ready";
  backend: string;
  camera: string;
  replay: string | null;
  map: string | null;
  mapAvailable: boolean;
  carCount: number;
  visibleCars: number[];
  time: number;
  frame: number;
  frameCount: number;
  playing: boolean;
  ended: boolean;
  speed: number;
  limitVisible: boolean;
  /** Centre of the view in simulator metres, and its scale. */
  view: { x: number; y: number; pixelsPerMetre: number };
  triangles: number;
  error: string | null;
  perf: PerfSummary;
}

function trajectoryBounds(replay: Replay): Bounds {
  const points: number[] = [];
  for (let car = 0; car < replay.carCount; car++) {
    const end = replay.inactiveFrom[car] ?? 0;
    for (let frame = 0; frame < end; frame++) {
      const base = (frame * replay.carCount + car) * 3;
      points.push(replay.poses[base] ?? 0, replay.poses[base + 1] ?? 0);
    }
  }
  if (points.length === 0) return { minX: -10, minY: -10, maxX: 10, maxY: 10 };
  const bounds = boundsOf(points);
  return {
    minX: bounds.minX - 3,
    minY: bounds.minY - 3,
    maxX: bounds.maxX + 3,
    maxY: bounds.maxY + 3,
  };
}

export class App {
  private session: Session | null = null;
  private phase: DashboardState["phase"] = "library";
  private lastError: string | null = null;
  private limitVisible = true;
  private lastFrameAt = 0;
  private benchmarking = false;
  private width = 1;
  private height = 1;

  private readonly aerial: AerialCamera;
  private readonly markers: CarMarkers;
  private readonly perf = new PerfMeter();
  private readonly timeline: Timeline;
  private readonly library: Library;
  private readonly pose: Pose = { x: 0, y: 0, heading: 0 };
  private readonly scratch = new THREE.Vector3();

  private readonly title = h("span", { class: "replay-title" }, "Aucun replay");
  private readonly subtitle = h("span", { class: "muted" });
  private readonly notices = h("div", { class: "notices" });
  private readonly loading = h("div", { class: "panel loading", hidden: true });
  private readonly dropHint = h("div", { class: "drop-hint", hidden: true }, "Déposez le replay");
  private readonly limitSwitch: HTMLButtonElement;
  private readonly bottom: HTMLElement;

  private constructor(
    private readonly stage: Stage,
    private readonly canvas: HTMLCanvasElement,
    hud: HTMLElement,
    markerLayer: HTMLElement,
  ) {
    this.aerial = new AerialCamera(canvas);
    this.markers = new CarMarkers(markerLayer);
    this.timeline = new Timeline({
      onToggle: () => this.session?.playback.toggle(),
      onSeek: (time) => this.session?.playback.seek(time),
      onSpeed: (speed) => {
        if (this.session) this.session.playback.speed = speed;
      },
    });
    this.library = new Library({
      onPickServerReplay: (name) => void this.openServerReplay(name),
      onPickFile: (file) => void this.openFile(file),
      onClose: () => this.library.hide(),
    });
    this.limitSwitch = h(
      "button",
      {
        class: "switch",
        type: "button",
        role: "switch",
        title: "Limite de DNF : bord de piste + 1 m (L)",
        onclick: () => this.setLimitVisible(!this.limitVisible),
      },
      h("span", { class: "switch-dot" }),
      "Limite de DNF",
    );
    this.bottom = h("div", { class: "hud-bottom", hidden: true }, this.timeline.element);

    hud.append(
      h(
        "div",
        { class: "hud-top" },
        h(
          "div",
          { class: "panel brand" },
          h("span", { class: "brand-mark" }, "Crash & Learn"),
          h("div", { class: "brand-replay" }, this.title, this.subtitle),
          h(
            "button",
            {
              class: "button",
              type: "button",
              title: "Ouvrir un replay (O)",
              onclick: () => this.library.show(this.session !== null),
            },
            icon(ICONS.folder, 14),
            "Ouvrir",
          ),
        ),
        h(
          "div",
          { class: "panel view-options" },
          h("span", { class: "camera-name", title: "Caméra" }, "Aérienne"),
          this.limitSwitch,
          h("span", { class: "badge", title: "Moteur de rendu" }, stage.backend),
        ),
      ),
      this.notices,
      this.perf.element,
      this.bottom,
      this.loading,
      this.library.element,
      this.dropHint,
    );
    this.setLimitVisible(true);
    this.bindWindow();
  }

  static async start(canvas: HTMLCanvasElement, hud: HTMLElement, markers: HTMLElement) {
    const query = new URLSearchParams(location.search);
    const stage = await Stage.create(canvas, query.get("renderer") === "webgl");
    const app = new App(stage, canvas, hud, markers);
    if (query.has("perf")) app.perf.toggle();
    app.resize();
    requestAnimationFrame((now) => app.frame(now));
    const replay = query.get("replay");
    if (replay) void app.openServerReplay(replay);
    else app.library.show(false);
    return app;
  }

  /** Snapshot for the browser tests and the performance measurements. */
  state(): DashboardState {
    const session = this.session;
    return {
      phase: this.phase,
      backend: this.stage.backend,
      camera: "Aérienne",
      replay: session?.replay.name ?? null,
      map: session?.replay.mapName ?? null,
      mapAvailable: session?.mapAvailable ?? false,
      carCount: session?.replay.carCount ?? 0,
      visibleCars: session
        ? session.cars.filter((car) => car.group.visible).map((car) => car.car)
        : [],
      time: session?.playback.time ?? 0,
      frame: session?.playback.frame ?? 0,
      frameCount: session?.replay.frameCount ?? 0,
      playing: session?.playback.playing ?? false,
      ended: session?.playback.ended ?? false,
      speed: session?.playback.speed ?? 1,
      limitVisible: this.limitVisible,
      view: {
        x: this.aerial.target.x,
        y: -this.aerial.target.z,
        pixelsPerMetre: this.aerial.pixelsPerMetre,
      },
      triangles: session?.track.triangles ?? 0,
      error: this.lastError,
      perf: this.perf.summary(),
    };
  }

  /** Centres the Aerial camera on a point, in simulator metres. For tests and screenshots. */
  focus(x: number, y: number, viewHeight: number): void {
    this.aerial.focus(x, y, viewHeight);
  }

  async openServerReplay(name: string): Promise<void> {
    this.beginLoading(`Chargement de ${name}`);
    try {
      const response = await fetch(`/api/replays/${encodeURIComponent(name)}`);
      if (!response.ok) {
        this.fail("Replay introuvable.", `Le serveur n'a pas de fichier « ${name} ».`);
        return;
      }
      await this.openReplay(name, new Uint8Array(await response.arrayBuffer()), true);
    } catch (error) {
      this.failUnexpected(error);
    }
  }

  async openFile(file: File): Promise<void> {
    this.beginLoading(`Chargement de ${file.name}`);
    try {
      await this.openReplay(file.name, new Uint8Array(await file.arrayBuffer()), false);
    } catch (error) {
      this.failUnexpected(error);
    }
  }

  private beginLoading(label: string): void {
    this.phase = "loading";
    this.lastError = null;
    this.library.setError(null);
    this.loading.hidden = false;
    setText(this.loading, label);
  }

  private fail(message: string, detail?: string): void {
    this.lastError = message;
    this.loading.hidden = true;
    this.phase = this.session ? "ready" : "library";
    this.library.show(this.session !== null);
    this.library.setError(message, detail);
  }

  private failUnexpected(error: unknown): void {
    console.error(error);
    this.fail(
      "Le replay n'a pas pu être ouvert.",
      error instanceof Error ? error.message : String(error),
    );
  }

  private async openReplay(name: string, bytes: Uint8Array, fromServer: boolean): Promise<void> {
    let replay: Replay;
    try {
      replay = parseReplay(name, bytes);
    } catch (error) {
      if (!(error instanceof ReplayError)) throw error;
      this.fail("Fichier de replay invalide.", error.message);
      return;
    }

    const notices: string[] = [];
    let track: TrackView | null = null;
    if (replay.mapName === null) {
      notices.push("Ce replay ne nomme pas sa carte : le circuit n'est pas affiché.");
    } else {
      try {
        const assets = await loadTrackAssets(replay.mapName);
        track = buildTrackView(assets, this.stage.maxAnisotropy);
      } catch (error) {
        if (!(error instanceof MapNotFoundError)) throw error;
        notices.push(`Carte « ${replay.mapName} » introuvable : le circuit n'est pas affiché.`);
      }
    }
    const mapAvailable = track !== null;
    track ??= buildFallbackGround(trajectoryBounds(replay));

    this.closeSession();
    const cars = Array.from({ length: replay.carCount }, (_, car) => new CarView(car));
    this.stage.simRoot.add(track.group, ...cars.map((car) => car.group));
    this.session = { replay, playback: new Playback(replay), track, cars, mapAvailable };
    track.limit.visible = this.limitVisible;

    this.markers.setCars(replay.carCount);
    this.timeline.setReplay(replay.duration, replay.frameCount);
    this.aerial.fit(track.bounds, HUD_TOP_PX, HUD_BOTTOM_PX);
    this.perf.reset();

    setText(this.title, name.replace(/\.npz$/, ""));
    const carLabel = replay.carCount === 1 ? "1 voiture" : `${replay.carCount} voitures`;
    setText(this.subtitle, `${replay.mapName ?? "Carte inconnue"} · ${carLabel}`);
    this.notices.replaceChildren(
      ...notices.map((text) => h("div", { class: "notice warning", role: "status" }, text)),
    );
    this.limitSwitch.disabled = !mapAvailable;
    this.bottom.hidden = false;
    this.loading.hidden = true;
    this.library.hide();
    this.phase = "ready";

    const url = new URL(location.href);
    if (fromServer) url.searchParams.set("replay", name);
    else url.searchParams.delete("replay");
    history.replaceState(null, "", url);
  }

  private closeSession(): void {
    const session = this.session;
    if (!session) return;
    this.stage.simRoot.remove(session.track.group, ...session.cars.map((car) => car.group));
    session.track.dispose();
    for (const car of session.cars) car.dispose();
    this.session = null;
  }

  private setLimitVisible(visible: boolean): void {
    this.limitVisible = visible;
    this.limitSwitch.setAttribute("aria-checked", String(visible));
    if (this.session) this.session.track.limit.visible = visible;
  }

  private resize(): void {
    this.width = Math.max(this.canvas.clientWidth, 1);
    this.height = Math.max(this.canvas.clientHeight, 1);
    this.stage.resize(this.width, this.height);
    this.aerial.resize(this.width, this.height);
  }

  private frame(now: number): void {
    requestAnimationFrame((next) => this.frame(next));
    const interval = this.lastFrameAt === 0 ? 16.7 : now - this.lastFrameAt;
    this.lastFrameAt = now;
    if (this.benchmarking) return;
    const started = performance.now();
    this.draw(interval);
    this.perf.sample(
      interval,
      performance.now() - started,
      now,
      `${this.stage.backend} · ${Math.round(this.session?.track.triangles ?? 0)} triangles`,
    );
  }

  /** Updates the scene for `interval` milliseconds of wall-clock time and renders it. */
  private draw(interval: number): void {
    const session = this.session;
    if (session) {
      const { playback, replay, cars } = session;
      // A long pause of the tab must not jump the race forward.
      playback.advance(Math.min(interval, 100) / 1000);
      const carPixels = CAR_LENGTH_M * this.aerial.pixelsPerMetre;
      for (const car of cars) {
        const active = samplePose(replay, playback.time, car.car, this.pose);
        car.group.visible = active;
        if (active) {
          car.setPose(this.pose.x, this.pose.y, this.pose.heading);
          const steering = replay.steering?.[playback.frame * replay.carCount + car.car];
          car.setSteering(steering ?? 0);
        }
        // Simulator (x, y) is (x, -z) in Three.js world axes.
        this.scratch.set(this.pose.x, 0.08, -this.pose.y);
        this.markers.place(
          car.car,
          active,
          this.scratch,
          this.aerial.camera,
          this.width,
          this.height,
          carPixels,
        );
      }
      this.timeline.update({
        time: playback.time,
        frame: playback.frame,
        playing: playback.playing,
        ended: playback.ended,
        speed: playback.speed,
      });
    }

    this.aerial.update();
    this.stage.fitShadow(this.aerial.target, this.aerial.viewRadius);
    this.stage.render(this.aerial.camera);
  }

  /**
   * Measures the cost of a frame without the display refresh limit: plays `frames` frames of
   * 1/60 s as fast as possible, and waits for the GPU to finish each one.
   */
  async benchmark(frames = 300): Promise<{ frames: number; msPerFrame: number; fps: number }> {
    const playback = this.session?.playback;
    if (!playback) throw new Error("open a replay before the benchmark");
    this.benchmarking = true;
    const wasPlaying = playback.playing;
    playback.playing = true;
    const started = performance.now();
    for (let i = 0; i < frames; i++) {
      if (playback.ended) playback.seek(0);
      playback.playing = true;
      this.draw(1000 / 60);
      // oxlint-disable-next-line no-await-in-loop -- one frame must end before the next starts
      await this.stage.finish();
    }
    const msPerFrame = (performance.now() - started) / frames;
    playback.playing = wasPlaying;
    this.benchmarking = false;
    return { frames, msPerFrame, fps: 1000 / msPerFrame };
  }

  private bindWindow(): void {
    new ResizeObserver(() => this.resize()).observe(this.canvas);
    this.canvas.addEventListener("dblclick", () => this.aerial.reset());

    window.addEventListener("keydown", (event) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === "Escape" && this.library.open && this.session) {
        this.library.hide();
        return;
      }
      if (this.library.open) return;
      const playback = this.session?.playback;
      const handlers: Record<string, () => void> = {
        " ": () => playback?.toggle(),
        ArrowRight: () => playback?.step(event.shiftKey ? 20 : 1),
        ArrowLeft: () => playback?.step(event.shiftKey ? -20 : -1),
        Home: () => playback?.seek(0),
        End: () => playback?.seek(playback.replay.duration),
        ArrowUp: () => this.shiftSpeed(1),
        ArrowDown: () => this.shiftSpeed(-1),
        l: () => this.setLimitVisible(!this.limitVisible),
        r: () => this.aerial.reset(),
        o: () => this.library.show(this.session !== null),
        p: () => this.perf.toggle(),
      };
      const handler = handlers[event.key.length === 1 ? event.key.toLowerCase() : event.key];
      if (!handler) return;
      event.preventDefault();
      handler();
    });

    let dragDepth = 0;
    window.addEventListener("dragenter", (event) => {
      event.preventDefault();
      dragDepth++;
      this.dropHint.hidden = false;
    });
    window.addEventListener("dragleave", () => {
      dragDepth = Math.max(dragDepth - 1, 0);
      if (dragDepth === 0) this.dropHint.hidden = true;
    });
    window.addEventListener("dragover", (event) => event.preventDefault());
    window.addEventListener("drop", (event) => {
      event.preventDefault();
      dragDepth = 0;
      this.dropHint.hidden = true;
      const file = event.dataTransfer?.files[0];
      if (file) void this.openFile(file);
    });
  }

  private shiftSpeed(direction: number): void {
    const playback = this.session?.playback;
    if (!playback) return;
    const index = SPEEDS.indexOf(playback.speed as (typeof SPEEDS)[number]);
    playback.speed = SPEEDS[Math.min(Math.max(index + direction, 0), SPEEDS.length - 1)] ?? 1;
  }
}
