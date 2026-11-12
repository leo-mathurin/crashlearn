/** The dashboard: opens a replay, builds its scene and runs the frame loop. */
import * as THREE from "three/webgpu";

import { lastActivePose, Playback, type Pose, samplePose, SPEEDS } from "./replay/playback";
import { parseReplay, type Replay, ReplayError } from "./replay/replay";
import { type CarValues, type Outcome, orderAt, outcomes } from "./replay/standings";
import { AerialCamera, type Insets } from "./scene/aerialCamera";
import { CAR_LENGTH_M, carColor, CarView } from "./scene/cars";
import { buildDnfMarker } from "./scene/dnfMarker";
import { LidarView } from "./scene/lidar";
import { Stage } from "./scene/stage";
import { buildFallbackGround, buildTrackView, type TrackView } from "./scene/trackView";
import { loadTrackAssets, MapNotFoundError } from "./track/assets";
import { type Bounds, boundsOf } from "./track/track";
import { h, icon, ICONS, setText } from "./ui/dom";
import { SceneLabels } from "./ui/labels";
import { Leaderboard } from "./ui/leaderboard";
import { Library } from "./ui/library";
import { PerfMeter, type PerfSummary } from "./ui/perf";
import { Telemetry } from "./ui/telemetry";
import { Timeline } from "./ui/timeline";

interface Session {
  replay: Replay;
  playback: Playback;
  track: TrackView;
  cars: CarView[];
  /** How the race of each car ended. */
  outcomes: Outcome[];
  /** Ground mark at the last pose of each car that ended in DNF, or null. */
  dnfMarkers: (THREE.Group | null)[];
  /** Null when the replay has no LiDAR ranges. */
  lidar: LidarView | null;
  /** False when the map of the replay was not found: the circuit is not drawn. */
  mapAvailable: boolean;
}

/** Space that the HUD covers on each side: the circuit is framed in what is left. */
const HUD_INSETS: Insets = { top: 70, bottom: 84, left: 330, right: 330 };
/** Below this width the side panels are over the scene: the circuit uses the full width. */
const SIDE_PANELS_MIN_WIDTH_PX = 1100;

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
  /** Position on screen of each car in simulator metres, or null for an inactive car. */
  carPositions: ({ x: number; y: number } | null)[];
  time: number;
  frame: number;
  frameCount: number;
  playing: boolean;
  ended: boolean;
  speed: number;
  limitVisible: boolean;
  lidarVisible: boolean;
  /** Voiture sélectionnée: the car whose telemetry and LiDAR are shown. */
  selectedCar: number;
  /** Cars from first to last at the playhead. */
  order: number[];
  /** Values of the selected car at the playhead. */
  telemetry: CarValues | null;
  /** Cars whose DNF marker is shown. */
  dnfMarkers: number[];
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
  private lidarVisible = true;
  private selectedCar = 0;
  private lastFrameAt = 0;
  private benchmarking = false;
  private width = 1;
  private height = 1;

  private readonly aerial: AerialCamera;
  private readonly labels: SceneLabels;
  private readonly leaderboard: Leaderboard;
  private readonly telemetry = new Telemetry();
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
  private readonly lidarSwitch: HTMLButtonElement;
  private readonly bottom: HTMLElement;

  private constructor(
    private readonly stage: Stage,
    private readonly canvas: HTMLCanvasElement,
    hud: HTMLElement,
    markerLayer: HTMLElement,
  ) {
    this.aerial = new AerialCamera(canvas);
    this.labels = new SceneLabels(markerLayer, (car) => this.selectCar(car));
    this.leaderboard = new Leaderboard((car) => this.selectCar(car));
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
    this.limitSwitch = toggleSwitch(
      "Limite de DNF",
      "Limite de DNF : bord de piste + 1 m (L)",
      () => this.setLimitVisible(!this.limitVisible),
    );
    this.lidarSwitch = toggleSwitch("LiDAR", "LiDAR de la voiture sélectionnée (I)", () =>
      this.setLidarVisible(!this.lidarVisible),
    );
    this.bottom = h("div", { class: "hud-bottom", hidden: true }, this.timeline.element);

    hud.append(
      h(
        "div",
        { class: "hud-column left" },
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
        this.leaderboard.element,
      ),
      h(
        "div",
        { class: "hud-column right" },
        h(
          "div",
          { class: "panel view-options" },
          h("span", { class: "camera-name", title: "Caméra" }, "Aérienne"),
          this.limitSwitch,
          this.lidarSwitch,
          h("span", { class: "badge", title: "Moteur de rendu" }, stage.backend),
        ),
        this.telemetry.element,
      ),
      this.notices,
      this.perf.element,
      this.bottom,
      this.loading,
      this.library.element,
      this.dropHint,
    );
    this.setLimitVisible(true);
    this.setLidarVisible(true);
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
      carPositions: session
        ? session.cars.map((car) =>
            car.group.visible ? { x: car.group.position.x, y: car.group.position.y } : null,
          )
        : [],
      time: session?.playback.time ?? 0,
      frame: session?.playback.frame ?? 0,
      frameCount: session?.replay.frameCount ?? 0,
      playing: session?.playback.playing ?? false,
      ended: session?.playback.ended ?? false,
      speed: session?.playback.speed ?? 1,
      limitVisible: this.limitVisible,
      lidarVisible: this.lidarVisible && Boolean(session?.lidar?.group.visible),
      selectedCar: this.selectedCar,
      order: session ? orderAt(session.replay, session.playback.frame) : [],
      telemetry: session ? this.telemetry.values(session.playback.time) : null,
      dnfMarkers: session
        ? session.dnfMarkers.flatMap((marker, car) => (marker?.visible ? [car] : []))
        : [],
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
    const results = outcomes(replay);
    const dnfMarkers = results.map((outcome, car) => {
      if (outcome.status !== "dnf" || !lastActivePose(replay, car, this.pose)) return null;
      return buildDnfMarker(this.pose.x, this.pose.y);
    });
    const lidar = replay.lidar ? new LidarView(replay.beamCount) : null;
    this.stage.simRoot.add(track.group, ...cars.map((car) => car.group));
    for (const object of [...dnfMarkers, lidar?.group]) {
      if (object) this.stage.simRoot.add(object);
    }
    this.session = {
      replay,
      playback: new Playback(replay),
      track,
      cars,
      outcomes: results,
      dnfMarkers,
      lidar,
      mapAvailable,
    };
    track.limit.visible = this.limitVisible;
    if (!replay.lidar) notices.push("Ce replay n'a pas de mesures LiDAR.");

    this.labels.setCars(replay.teamNames);
    this.leaderboard.setReplay(replay);
    this.telemetry.setReplay(replay);
    this.selectCar(0);
    this.lidarSwitch.disabled = lidar === null;
    this.timeline.setReplay(replay.duration, replay.frameCount);
    this.timeline.setEvents(
      results.flatMap((outcome, car) =>
        outcome.status === "active"
          ? []
          : [
              {
                time: outcome.frame * replay.simDt,
                color: carColor(car),
                label: `${replay.teamNames[car]} : ${outcome.status === "dnf" ? "DNF" : "arrivée"}`,
              },
            ],
      ),
    );
    const wide = this.width >= SIDE_PANELS_MIN_WIDTH_PX;
    this.aerial.fit(track.bounds, wide ? HUD_INSETS : { ...HUD_INSETS, left: 0, right: 0 });
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
    for (const marker of session.dnfMarkers) {
      if (marker) this.stage.simRoot.remove(marker);
    }
    if (session.lidar) {
      this.stage.simRoot.remove(session.lidar.group);
      session.lidar.dispose();
    }
    session.track.dispose();
    for (const car of session.cars) car.dispose();
    this.session = null;
  }

  /** Makes `car` the Voiture sélectionnée: its telemetry and its LiDAR are shown. */
  selectCar(car: number): void {
    if (!this.session || car < 0 || car >= this.session.replay.carCount) return;
    this.selectedCar = car;
    this.labels.setSelected(car);
    this.leaderboard.setSelected(car);
    this.telemetry.setCar(car);
  }

  private setLidarVisible(visible: boolean): void {
    this.lidarVisible = visible;
    this.lidarSwitch.setAttribute("aria-checked", String(visible));
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
    // A long pause of the tab must not jump the race forward.
    session?.playback.advance(Math.min(interval, 100) / 1000);
    if (session) this.placeCars(session);

    this.aerial.update();
    const camera = this.aerial.camera;
    camera.updateMatrixWorld();
    if (session) this.updateHud(session, camera);

    this.stage.fitShadow(this.aerial.target, this.aerial.viewRadius);
    this.stage.render(camera);
  }

  /** Moves the cars, the LiDAR and the DNF marks to the playhead. */
  private placeCars(session: Session): void {
    const { playback, replay, cars, lidar, dnfMarkers, outcomes: results } = session;
    const frame = playback.frame;
    for (const car of cars) {
      const active = samplePose(replay, playback.time, car.car, this.pose);
      car.group.visible = active;
      if (!active) continue;
      car.setPose(this.pose.x, this.pose.y, this.pose.heading);
      car.setSteering(replay.steering?.[frame * replay.carCount + car.car] ?? 0);
      if (lidar && car.car === this.selectedCar) {
        lidar.update(replay, frame, playback.time / replay.simDt - frame, car.car, this.pose);
      }
    }
    if (lidar) {
      lidar.group.visible = this.lidarVisible && (cars[this.selectedCar]?.group.visible ?? false);
    }
    dnfMarkers.forEach((marker, car) => {
      // The mark appears at the frame where the simulator declares the DNF.
      if (marker) marker.visible = frame >= (results[car]?.frame ?? Infinity);
    });
  }

  /** Updates the screen labels and the panels for the playhead and the camera. */
  private updateHud(session: Session, camera: THREE.Camera): void {
    const { playback, cars, dnfMarkers } = session;
    const carPixels = CAR_LENGTH_M * this.aerial.pixelsPerMetre;
    for (const car of cars) {
      // Simulator (x, y, z) is (x, z, -y) in Three.js world axes.
      this.scratch.set(car.group.position.x, 0.08, -car.group.position.y);
      this.labels.placeCar(
        car.car,
        car.group.visible,
        this.scratch,
        camera,
        this.width,
        this.height,
        carPixels,
      );
      const marker = dnfMarkers[car.car];
      if (!marker) continue;
      this.scratch.set(marker.position.x, 0.08, -marker.position.y);
      this.labels.placeDnf(car.car, marker.visible, this.scratch, camera, this.width, this.height);
    }
    this.leaderboard.update(playback.frame);
    this.telemetry.update(playback.time);
    this.timeline.update({
      time: playback.time,
      frame: playback.frame,
      playing: playback.playing,
      ended: playback.ended,
      speed: playback.speed,
    });
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
        i: () => this.setLidarVisible(!this.lidarVisible),
        "1": () => this.selectCar(0),
        "2": () => this.selectCar(1),
        "3": () => this.selectCar(2),
        "4": () => this.selectCar(3),
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

function toggleSwitch(label: string, title: string, onToggle: () => void): HTMLButtonElement {
  return h(
    "button",
    { class: "switch", type: "button", role: "switch", title, onclick: onToggle },
    h("span", { class: "switch-dot" }),
    label,
  );
}
