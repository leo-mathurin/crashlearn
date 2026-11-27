/** The dashboard: opens a replay, builds its scene and runs the frame loop. */
import * as THREE from "three/webgpu";

import { FollowTracker } from "./replay/follow";
import {
  isActive,
  lastActivePose,
  Playback,
  type Pose,
  sampleChannel,
  samplePose,
  SPEEDS,
} from "./replay/playback";
import { parseReplay, type Replay, ReplayError } from "./replay/replay";
import { type CarValues, type Outcome, orderAt, outcomes } from "./replay/standings";
import type { Insets } from "./scene/aerialCamera";
import { CAMERA_LABELS, CameraDirector, type CameraMode } from "./scene/cameraDirector";
import { CAR_LENGTH_M, carColor, CarView } from "./scene/cars";
import { buildDecor, type Decor } from "./scene/decor";
import { buildDnfMarker } from "./scene/dnfMarker";
import { LidarView } from "./scene/lidar";
import { Stage } from "./scene/stage";
import { asphalt, concrete, grass } from "./scene/textures";
import { buildFallbackGround, buildTrackView, type TrackView } from "./scene/trackView";
import { loadTrackAssets, MapNotFoundError } from "./track/assets";
import { type DecorLayout, layoutDecor } from "./track/decorLayout";
import { type Bounds, boundsOf } from "./track/track";
import { CameraSwitch, FollowMenu } from "./ui/cameraControls";
import { h, icon, ICONS, setText } from "./ui/dom";
import { SceneLabels } from "./ui/labels";
import { Leaderboard } from "./ui/leaderboard";
import { Library } from "./ui/library";
import { MiniMap, type MiniMapCar } from "./ui/minimap";
import { PerfMeter, type PerfSummary } from "./ui/perf";
import { Telemetry } from "./ui/telemetry";
import { Timeline } from "./ui/timeline";

interface Session {
  replay: Replay;
  playback: Playback;
  track: TrackView;
  cars: CarView[];
  /** Everything that moves or belongs to the race: cars, LiDAR, DNF marks. */
  actors: THREE.Group;
  /** Optional scenery around the circuit. */
  decor: Decor;
  /** How the race of each car ended. */
  outcomes: Outcome[];
  /** Ground mark at the last pose of each car that ended in DNF, or null. */
  dnfMarkers: (THREE.Group | null)[];
  /** Null when the replay has no LiDAR ranges. */
  lidar: LidarView | null;
  /** False when the map of the replay was not found: the circuit is not drawn. */
  mapAvailable: boolean;
}

/** Margin between the circuit and the HUD panels around it, in pixels. */
const HUD_MARGIN_PX = 14;
/** Below this width the side panels are over the scene: the circuit uses the full width. */
const SIDE_PANELS_MIN_WIDTH_PX = 1100;

/**
 * Frame time above which the scene is drawn with fewer pixels, and the steps of resolution. A
 * frame of 24 ms is 42 frames per second: under that, motion is not fluid any more.
 */
const SLOW_FRAME_MS = 24;
const RESOLUTION_STEPS = [1, 0.8, 0.65, 0.5] as const;
/** Frames in one measurement of the frame time, and slow measurements before a step down. */
const PACE_WINDOW = 45;
const SLOW_WINDOWS = 2;

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
  /** True when the optional scenery is drawn. */
  decorVisible: boolean;
  decorTriangles: number;
  lidarVisible: boolean;
  /** Voiture sélectionnée: the car whose telemetry and LiDAR are shown. */
  selectedCar: number;
  /** Voiture suivie: the car that the Chase camera follows. */
  followedCar: number;
  /** True while the followed car is the selected car. */
  followLinked: boolean;
  /** Share of the full resolution that the scene is drawn at. */
  resolutionScale: number;
  /** True during the move from one camera to another. */
  cameraMoving: boolean;
  /** Position of the camera in simulator metres (z up). */
  cameraPosition: { x: number; y: number; z: number };
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
  /** Time of each step of the last opening of a replay, in milliseconds. */
  loadMs: Record<string, number>;
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
  private loadMs: Record<string, number> = {};
  private loadMark = 0;
  /** Number of the last replay opening: an older one that ends late is dropped. */
  private loadId = 0;
  /** True while the GPU programs of a new scene build: nothing is rendered. */
  private preparing = 0;
  private paceSum = 0;
  private paceCount = 0;
  private slowWindows = 0;
  private limitVisible = true;
  private lidarVisible = true;
  private decorVisible = readPreference(DECOR_PREFERENCE, true);
  private selectedCar = 0;
  private lastFrameAt = 0;
  private benchmarking = false;
  private width = 1;
  private height = 1;

  private readonly cameras: CameraDirector;
  private readonly follow = new FollowTracker();
  private readonly minimap = new MiniMap();
  private readonly cameraSwitch: CameraSwitch;
  private readonly followMenu: FollowMenu;
  /** Playback time at the previous frame, to see a seek. */
  private lastPlaybackTime = 0;
  private readonly followedPose: Pose = { x: 0, y: 0, heading: 0 };
  private readonly focusScratch = new THREE.Vector3();
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
  private readonly decorSwitch: HTMLButtonElement;
  private readonly bottom: HTMLElement;
  private readonly top: HTMLElement;

  private constructor(
    private readonly stage: Stage,
    private readonly canvas: HTMLCanvasElement,
    hud: HTMLElement,
    markerLayer: HTMLElement,
  ) {
    this.cameras = new CameraDirector(canvas);
    this.cameraSwitch = new CameraSwitch((mode) => this.setCamera(mode, true));
    this.followMenu = new FollowMenu({
      onLink: () => this.setFollowed(this.selectedCar, true),
      onFollow: (car) => this.setFollowed(car, false),
    });
    this.cameraSwitch.setMode("aerial");
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
    this.decorSwitch = toggleSwitch("Décor", "Décor : arbres, tribunes, portique, ciel (G)", () =>
      this.setDecorVisible(!this.decorVisible),
    );
    this.decorSwitch.setAttribute("aria-checked", String(this.decorVisible));
    this.lidarSwitch = toggleSwitch("LiDAR", "LiDAR de la voiture sélectionnée (I)", () =>
      this.setLidarVisible(!this.lidarVisible),
    );
    this.bottom = h("div", { class: "hud-bottom", hidden: true }, this.timeline.element);

    this.top = h(
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
        this.cameraSwitch.element,
        this.followMenu.element,
        this.limitSwitch,
        this.lidarSwitch,
        this.decorSwitch,
        h("span", { class: "badge", title: "Moteur de rendu" }, stage.backend),
      ),
    );
    hud.append(
      this.top,
      this.notices,
      h(
        "div",
        { class: "hud-middle" },
        h("div", { class: "hud-side left" }, this.leaderboard.element, this.perf.element),
        h("div", { class: "hud-side right" }, this.telemetry.element, this.minimap.element),
      ),
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
    // The surface textures are drawn while the user chooses a replay.
    setTimeout(() => {
      for (const draw of [asphalt, grass, concrete]) draw(stage.maxAnisotropy);
    }, 0);
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
      camera: CAMERA_LABELS[this.cameras.mode],
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
      decorVisible: this.decorVisible,
      decorTriangles: session?.decor.triangles ?? 0,
      lidarVisible: this.lidarVisible && Boolean(session?.lidar?.group.visible),
      selectedCar: this.selectedCar,
      followedCar: this.follow.followed,
      followLinked: this.follow.linked,
      resolutionScale: this.stage.resolutionScale,
      cameraMoving: this.cameras.inTransition,
      cameraPosition: {
        x: this.cameras.camera.position.x,
        y: -this.cameras.camera.position.z,
        z: this.cameras.camera.position.y,
      },
      order: session ? orderAt(session.replay, session.playback.frame) : [],
      telemetry: session ? this.telemetry.values(session.playback.time) : null,
      dnfMarkers: session
        ? session.dnfMarkers.flatMap((marker, car) => (marker?.visible ? [car] : []))
        : [],
      view: {
        x: this.cameras.focusPoint(this.focusScratch).x,
        y: -this.cameras.focusPoint(this.focusScratch).z,
        pixelsPerMetre: this.cameras.pixelsPerMetreAt(this.cameras.focusPoint(this.focusScratch)),
      },
      triangles: this.triangles(),
      error: this.lastError,
      perf: this.perf.summary(),
      loadMs: this.loadMs,
    };
  }

  /** Centres the Aerial camera on a point, in simulator metres. For tests and screenshots. */
  focus(x: number, y: number, viewHeight: number): void {
    this.setCamera("aerial", false);
    this.cameras.aerial.focus(x, y, viewHeight);
  }

  async openServerReplay(name: string): Promise<void> {
    this.beginLoading(`Chargement de ${name}`);
    try {
      const response = await fetch(`/api/replays/${encodeURIComponent(name)}`);
      if (!response.ok) {
        this.fail("Replay introuvable.", `Le serveur n'a pas de fichier « ${name} ».`);
        return;
      }
      const bytes = new Uint8Array(await response.arrayBuffer());
      this.lap("download");
      await this.openReplay(name, bytes, true);
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

  /** Records the time since the previous step of the opening of a replay. */
  private lap(step: string): void {
    const now = performance.now();
    this.loadMs[step] = Math.round(now - this.loadMark);
    this.loadMark = now;
  }

  private beginLoading(label: string): void {
    this.loadId++;
    this.loadMs = {};
    this.loadMark = performance.now();
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
    const loadId = this.loadId;
    let replay: Replay;
    try {
      replay = parseReplay(name, bytes);
    } catch (error) {
      if (!(error instanceof ReplayError)) throw error;
      this.fail("Fichier de replay invalide.", error.message);
      return;
    }
    this.lap("parse");

    const notices: string[] = [];
    let track: TrackView | null = null;
    let layout: DecorLayout | null = null;
    if (replay.mapName === null) {
      notices.push("Ce replay ne nomme pas sa carte : le circuit n'est pas affiché.");
    } else {
      try {
        const assets = await loadTrackAssets(replay.mapName);
        if (loadId !== this.loadId) return;
        this.lap("map");
        track = buildTrackView(assets, this.stage.maxAnisotropy);
        this.lap("track");
        layout = layoutDecor(assets.centerline, assets.name);
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
    const actors = new THREE.Group();
    actors.add(...cars.map((car) => car.group));
    for (const object of [...dnfMarkers, lidar?.group]) {
      if (object) actors.add(object);
    }
    const decor = buildDecor(layout, track.bounds, replay.frameCount);
    decor.setVisible(this.decorVisible);
    this.lap("decor");
    this.stage.simRoot.add(track.group, decor.ground, actors);
    this.stage.scene.add(decor.sky);
    this.session = {
      replay,
      playback: new Playback(replay),
      track,
      cars,
      actors,
      decor,
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
    this.followMenu.setCars(replay.teamNames);
    this.follow.linked = true;
    this.selectCar(0);
    this.lastPlaybackTime = 0;
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
    this.minimap.setBounds(track.bounds);
    this.perf.reset();

    setText(this.title, name.replace(/\.npz$/, ""));
    const carLabel = replay.carCount === 1 ? "1 voiture" : `${replay.carCount} voitures`;
    setText(this.subtitle, `${replay.mapName ?? "Carte inconnue"} · ${carLabel}`);
    this.notices.replaceChildren(
      ...notices.map((text) => h("div", { class: "notice warning", role: "status" }, text)),
    );
    this.limitSwitch.disabled = !mapAvailable;
    this.bottom.hidden = false;
    this.library.hide();
    // The panels now have their size: the circuit is framed in the space between them.
    this.cameras.fit(track.bounds, this.hudInsets());
    this.lap("hud");
    this.placeCars(this.session);
    this.preparing++;
    try {
      await this.stage.warmUp([this.cameras.aerial.camera, this.cameras.free.camera]);
    } finally {
      this.preparing--;
    }
    if (loadId !== this.loadId) return;
    this.lap("programs");
    // The first frame sends the textures and the geometry to the GPU: it is part of the wait.
    this.draw(0);
    await this.stage.finish();
    this.lap("firstFrame");
    this.perf.setLoad(this.loadMs);
    this.loading.hidden = true;
    this.phase = "ready";

    const url = new URL(location.href);
    if (fromServer) url.searchParams.set("replay", name);
    else url.searchParams.delete("replay");
    history.replaceState(null, "", url);
  }

  private closeSession(): void {
    const session = this.session;
    if (!session) return;
    this.stage.simRoot.remove(session.track.group, session.actors);
    session.lidar?.dispose();
    this.stage.simRoot.remove(session.decor.ground);
    this.stage.scene.remove(session.decor.sky);
    session.decor.dispose();
    session.track.dispose();
    for (const car of session.cars) car.dispose();
    this.session = null;
  }

  /** Space that the HUD panels cover on each side of the screen, in pixels. */
  private hudInsets(): Insets {
    const edge = (element: HTMLElement, side: "left" | "right" | "top" | "bottom"): number => {
      const box = element.getBoundingClientRect();
      if (box.width === 0) return 0;
      if (side === "left" || side === "top") return box[side === "left" ? "right" : "bottom"];
      return (side === "right" ? this.width : this.height) - box[side === "right" ? "left" : "top"];
    };
    const wide = this.width >= SIDE_PANELS_MIN_WIDTH_PX;
    return {
      top: edge(this.top, "top") + HUD_MARGIN_PX,
      bottom: edge(this.bottom, "bottom") + HUD_MARGIN_PX,
      left: wide ? edge(this.leaderboard.element, "left") + HUD_MARGIN_PX : 0,
      right: wide ? edge(this.telemetry.element, "right") + HUD_MARGIN_PX : 0,
    };
  }

  /** Triangles of the circuit, and of the scenery when it is drawn. */
  private triangles(): number {
    const session = this.session;
    if (!session) return 0;
    return session.track.triangles + (this.decorVisible ? session.decor.triangles : 0);
  }

  /** Makes `car` the Voiture sélectionnée: its telemetry and its LiDAR are shown. */
  selectCar(car: number): void {
    if (!this.session || car < 0 || car >= this.session.replay.carCount) return;
    this.selectedCar = car;
    this.labels.setSelected(car);
    this.leaderboard.setSelected(car);
    this.telemetry.setCar(car);
    if (this.follow.linked) this.setFollowed(car, true);
  }

  /** Chooses the camera. With `animate`, the view glides to it. */
  setCamera(mode: CameraMode, animate: boolean): void {
    this.cameras.setMode(mode, animate);
    this.cameraSwitch.setMode(mode);
    this.followMenu.element.hidden = mode !== "chase";
  }

  /**
   * Makes `car` the Voiture suivie. `linked`: it follows the selected car from now on.
   * If not, it stays on this car when the selection changes.
   */
  setFollowed(car: number, linked: boolean): void {
    const session = this.session;
    if (!session || car < 0 || car >= session.replay.carCount) return;
    this.follow.linked = linked;
    this.follow.follow(car, isActive(session.replay, session.playback.frame, car));
    this.followMenu.setFollowed(car, linked);
  }

  private setLidarVisible(visible: boolean): void {
    this.lidarVisible = visible;
    this.lidarSwitch.setAttribute("aria-checked", String(visible));
  }

  /** Shows or hides the optional scenery. The choice is kept for the next visit. */
  setDecorVisible(visible: boolean): void {
    this.decorVisible = visible;
    this.decorSwitch.setAttribute("aria-checked", String(visible));
    this.session?.decor.setVisible(visible);
    writePreference(DECOR_PREFERENCE, visible);
    // The mini-map is a picture of the scene: take it again.
    this.minimap.stale = true;
  }

  private setLimitVisible(visible: boolean): void {
    this.limitVisible = visible;
    this.limitSwitch.setAttribute("aria-checked", String(visible));
    if (this.session) this.session.track.limit.visible = visible;
    // The mini-map is a picture of the circuit: take it again.
    this.minimap.stale = true;
  }

  private resize(): void {
    this.width = Math.max(this.canvas.clientWidth, 1);
    this.height = Math.max(this.canvas.clientHeight, 1);
    this.stage.resize(this.width, this.height);
    this.cameras.resize(this.width, this.height);
  }

  private frame(now: number): void {
    requestAnimationFrame((next) => this.frame(next));
    const interval = this.lastFrameAt === 0 ? 16.7 : now - this.lastFrameAt;
    this.lastFrameAt = now;
    if (this.benchmarking || this.preparing > 0) return;
    const started = performance.now();
    this.draw(interval);
    const scale = this.stage.resolutionScale;
    this.perf.sample(
      interval,
      performance.now() - started,
      now,
      `${this.stage.backend} · ${Math.round(this.triangles())} triangles` +
        (scale < 1 ? ` · rendu à ${Math.round(scale * 100)} %` : ""),
    );
    this.watchPace(interval);
  }

  /**
   * Lowers the resolution of the scene when the GPU cannot keep the pace: fewer pixels, but a
   * fluid motion. The resolution does not go up again during the visit, so that it cannot
   * change back and forth.
   */
  private watchPace(interval: number): void {
    // A long interval is a hidden tab or a stop of the page, not a slow GPU.
    if (!this.session || document.hidden || interval > 250) return;
    this.paceSum += interval;
    if (++this.paceCount < PACE_WINDOW) return;
    const mean = this.paceSum / this.paceCount;
    this.paceSum = 0;
    this.paceCount = 0;
    this.slowWindows = mean > SLOW_FRAME_MS ? this.slowWindows + 1 : 0;
    if (this.slowWindows < SLOW_WINDOWS) return;
    this.slowWindows = 0;
    const step = RESOLUTION_STEPS.indexOf(this.stage.resolutionScale as 1);
    const next = RESOLUTION_STEPS[step + 1];
    if (next === undefined) return;
    this.stage.setResolutionScale(next, this.width, this.height);
    this.minimap.stale = true;
  }

  /** Updates the scene for `interval` milliseconds of wall-clock time and renders it. */
  private draw(interval: number): void {
    const session = this.session;
    // A long pause of the tab must not jump the race forward.
    session?.playback.advance(Math.min(interval, 100) / 1000);
    if (session) this.placeCars(session);

    const seconds = Math.min(interval, 100) / 1000;
    let followed: Pose | null = null;
    let seeked = false;
    if (session) {
      const { playback } = session;
      // More time than the frame can explain, or time going back: the user moved the playhead.
      const advance = playback.time - this.lastPlaybackTime;
      seeked = advance < 0 || advance > seconds * playback.speed + 0.3;
      this.lastPlaybackTime = playback.time;
      followed = this.followCar(session);
    }
    this.cameras.update(seconds, followed, seeked);
    const camera = this.cameras.camera;
    camera.updateMatrixWorld();
    if (session) {
      this.updateHud(session, camera);
      if (this.minimap.stale) this.takeMiniMapPicture(session);
      this.drawMiniMap(session);
    }

    this.stage.fitShadow(this.focusScratch, this.cameras.shadowFocus(this.focusScratch));
    this.stage.setFog(this.cameras.focusDistance());
    this.stage.render(camera, this.decorVisible);
  }

  /**
   * Applies the rule of the Voiture suivie and returns the pose that the Chase camera follows:
   * the car while it is active, its last pose after that.
   */
  private followCar(session: Session): Pose | null {
    const { replay, playback, cars } = session;
    const frame = playback.frame;
    const active = cars[this.follow.followed]?.group.visible ?? false;
    const next = this.follow.update(performance.now(), active, () => {
      const best = orderAt(replay, frame).find((car) => isActive(replay, frame, car));
      return best ?? null;
    });
    if (next !== null) {
      // Linked: the selected car moves with the followed car, they stay the same car.
      if (this.follow.linked) this.selectCar(next);
      else this.setFollowed(next, false);
    }
    const car = cars[this.follow.followed];
    if (car?.group.visible) {
      this.followedPose.x = car.group.position.x;
      this.followedPose.y = car.group.position.y;
      this.followedPose.heading = car.group.rotation.z;
      return this.followedPose;
    }
    return lastActivePose(replay, this.follow.followed, this.followedPose)
      ? this.followedPose
      : null;
  }

  /** Renders the circuit from above, without the cars, and keeps it as the mini-map picture. */
  private takeMiniMapPicture(session: Session): void {
    const { minimap } = this;
    session.actors.visible = false;
    minimap.prepare(this.canvas.width, this.canvas.height);
    this.stage.fitShadow(minimap.center(this.focusScratch), minimap.radius);
    this.stage.setFog(minimap.camera.position.y);
    this.stage.render(minimap.camera, this.decorVisible);
    minimap.capture(this.canvas);
    session.actors.visible = true;
  }

  private drawMiniMap(session: Session): void {
    const cars: (MiniMapCar | null)[] = session.cars.map((car) =>
      car.group.visible
        ? {
            x: car.group.position.x,
            y: car.group.position.y,
            color: carColor(car.car),
            selected: car.car === this.selectedCar,
          }
        : null,
    );
    const marks = session.dnfMarkers.flatMap((marker) =>
      marker?.visible ? [{ x: marker.position.x, y: marker.position.y }] : [],
    );
    const focus = this.cameras.focusPoint(this.focusScratch);
    this.minimap.draw(cars, marks, { x: focus.x, y: -focus.z });
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
      car.setSteering(
        replay.steering ? sampleChannel(replay, replay.steering, playback.time, car.car) : 0,
      );
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
    for (const car of cars) {
      // Simulator (x, y, z) is (x, z, -y) in Three.js world axes.
      this.scratch.set(car.group.position.x, 0.08, -car.group.position.y);
      const carPixels = CAR_LENGTH_M * this.cameras.pixelsPerMetreAt(this.scratch);
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

    window.addEventListener("keydown", (event) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key === "Escape" && this.library.open && this.session) {
        this.library.hide();
        return;
      }
      if (this.library.open) return;
      const target = event.target;
      if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) return;
      // In the Free camera, the move keys move the camera: they are not shortcuts there.
      if (this.cameras.mode === "free" && this.cameras.free.pressKey(event.code)) {
        this.cameras.free.fast = event.shiftKey;
        event.preventDefault();
        return;
      }
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
        g: () => this.setDecorVisible(!this.decorVisible),
        "1": () => this.selectCar(0),
        "2": () => this.selectCar(1),
        "3": () => this.selectCar(2),
        "4": () => this.selectCar(3),
        r: () => this.cameras.reset(),
        // A camera chosen with the keyboard changes at once: no motion to wait for.
        a: () => this.setCamera("aerial", false),
        f: () => this.setCamera("free", false),
        c: () => this.setCamera("chase", false),
        o: () => this.library.show(this.session !== null),
        p: () => this.perf.toggle(),
      };
      const handler = handlers[event.key.length === 1 ? event.key.toLowerCase() : event.key];
      if (!handler) return;
      event.preventDefault();
      handler();
    });

    window.addEventListener("keyup", (event) => {
      this.cameras.free.releaseKey(event.code);
      this.cameras.free.fast = event.shiftKey;
    });
    // A key that goes up while the window has no focus gives no event: stop the move.
    window.addEventListener("blur", () => this.cameras.free.releaseKeys());

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

const DECOR_PREFERENCE = "dashboard.decor";

/** Reads a yes or no choice kept in the browser. Storage can be refused: use the default. */
function readPreference(key: string, fallback: boolean): boolean {
  try {
    const stored = localStorage.getItem(key);
    return stored === null ? fallback : stored === "1";
  } catch {
    return fallback;
  }
}

function writePreference(key: string, value: boolean): void {
  try {
    localStorage.setItem(key, value ? "1" : "0");
  } catch {
    // The choice then lasts for this visit only.
  }
}
