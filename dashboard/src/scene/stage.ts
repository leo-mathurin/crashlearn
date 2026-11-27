/**
 * Renderer, scene and lights. The renderer is the WebGPU renderer of Three.js: it uses WebGPU
 * when the browser has it and WebGL 2 if not, with the same scene code.
 */
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import * as THREE from "three/webgpu";

export type Backend = "WebGPU" | "WebGL2";

const SHADOW_MAP_SIZE = 2048;
/** Direction to the sun, in Three.js world axes (y up): a low afternoon sun from the south-west. */
const SUN_DIRECTION = new THREE.Vector3(-0.55, 0.62, 0.56).normalize();

export const HAZE_COLOR = 0xa9bdd0;

/**
 * Layer of the ground and of the optional scenery. Everything else (track, walls, cars, LiDAR)
 * is on the default layer 0.
 */
export const SCENERY_LAYER = 1;

const ALL_LAYERS = 1 | (1 << SCENERY_LAYER);

export class Stage {
  readonly scene = new THREE.Scene();
  /**
   * Root for everything authored in simulator coordinates (x east, y north, z up). Three.js has
   * y up: this group turns the simulator axes to (x, z, -y).
   */
  readonly simRoot = new THREE.Group();
  private readonly sun = new THREE.DirectionalLight(0xfff1dc, 2.6);
  private readonly fog = new THREE.Fog(HAZE_COLOR, 1e5, 1e6);
  private scale = 1;

  private constructor(
    readonly renderer: THREE.WebGPURenderer,
    readonly backend: Backend,
  ) {
    this.simRoot.rotation.x = -Math.PI / 2;
    this.scene.add(this.simRoot);
    // Sky haze behind the ground, for the cameras that see the horizon.
    this.scene.background = new THREE.Color(HAZE_COLOR);
    this.scene.fog = this.fog;

    const ambient = new THREE.HemisphereLight(0xcfe2ff, 0x4d5a3a, 0.9);
    this.scene.add(ambient);
    // The lights shine on both layers, and the shadow map holds the objects of both: a wall
    // still casts its shadow on the grass when the two layers are drawn one after the other.
    for (const object of [ambient, this.sun, this.sun.shadow.camera]) {
      object.layers.enable(SCENERY_LAYER);
    }
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.02;
    this.scene.add(this.sun, this.sun.target);

    const pmrem = new THREE.PMREMGenerator(renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.45;
    pmrem.dispose();
  }

  /** Creates the renderer on a canvas. `forceWebGL` skips WebGPU, for tests and comparisons. */
  static async create(canvas: HTMLCanvasElement, forceWebGL: boolean): Promise<Stage> {
    const renderer = new THREE.WebGPURenderer({ canvas, antialias: true, forceWebGL });
    await renderer.init();
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.toneMapping = THREE.NeutralToneMapping;
    renderer.toneMappingExposure = 0.95;
    const isWebGPU = (renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend === true;
    return new Stage(renderer, isWebGPU ? "WebGPU" : "WebGL2");
  }

  get maxAnisotropy(): number {
    return Math.min(this.renderer.getMaxAnisotropy(), 8);
  }

  resize(width: number, height: number): void {
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2) * this.scale);
    this.renderer.setSize(width, height, false);
  }

  /** Share of the full resolution that the scene is drawn at, from 0 to 1. */
  get resolutionScale(): number {
    return this.scale;
  }

  /**
   * Draws the scene with fewer pixels, for a GPU that cannot keep the frame rate at the full
   * resolution of the screen. The HUD is not part of the scene: it stays sharp.
   */
  setResolutionScale(scale: number, width: number, height: number): void {
    this.scale = scale;
    this.resize(width, height);
  }

  /**
   * Prepares the GPU programs of everything in the scene, for each camera of `cameras`: also
   * what is hidden or out of view now. Without this, each object stops the frame loop the first
   * time it comes on screen (a change of camera, the first DNF mark), for as long as its
   * program takes to build. The programs build in parallel where the browser can do it.
   * Nothing must be rendered before the promise resolves.
   */
  async warmUp(cameras: readonly THREE.Camera[]): Promise<void> {
    const hidden: THREE.Object3D[] = [];
    const culled: THREE.Object3D[] = [];
    this.scene.traverse((object) => {
      if (!object.visible) hidden.push(object);
      if (object.frustumCulled) culled.push(object);
      object.visible = true;
      object.frustumCulled = false;
    });
    try {
      for (const camera of cameras) {
        const { mask } = camera.layers;
        // The scenery is drawn in a pass of its own, or with the race in one pass.
        for (const layers of [1 << SCENERY_LAYER, 1, ALL_LAYERS]) {
          camera.layers.mask = layers;
          // oxlint-disable-next-line no-await-in-loop -- one render state at a time
          await this.renderer.compileAsync(this.scene, camera);
        }
        camera.layers.mask = mask;
      }
    } finally {
      for (const object of hidden) object.visible = false;
      for (const object of culled) object.frustumCulled = true;
    }
  }

  /**
   * Points the shadow map at the part of the world on screen: a disc of `radius` metres around
   * `center` (Three.js world axes). The centre moves by whole shadow texels, so that shadows do
   * not shimmer when the view moves.
   */
  fitShadow(center: THREE.Vector3, radius: number): void {
    const size = Math.min(Math.max(radius, 4), 120);
    const texel = (size * 2) / SHADOW_MAP_SIZE;
    const x = Math.round(center.x / texel) * texel;
    const z = Math.round(center.z / texel) * texel;
    this.sun.target.position.set(x, 0, z);
    this.sun.position.set(x, 0, z).addScaledVector(SUN_DIRECTION, 300);
    const camera = this.sun.shadow.camera;
    camera.left = -size;
    camera.right = size;
    camera.top = size;
    camera.bottom = -size;
    camera.near = 300 - size * 1.5 - 5;
    camera.far = 300 + size * 1.5 + 5;
    camera.updateProjectionMatrix();
  }

  /**
   * Fades the far ground into the haze. `focusDistance` is the distance from the camera to the
   * point it looks at: the fog starts beyond it, so the subject of the view is always clear,
   * also for the Aerial camera that is high above the ground.
   */
  setFog(focusDistance: number): void {
    this.fog.near = focusDistance + 140;
    this.fog.far = focusDistance + 900;
  }

  /** Resolves when the GPU has finished the frames submitted so far. For measurements. */
  async finish(): Promise<void> {
    const backend = this.renderer.backend as {
      device?: GPUDevice;
      gl?: WebGL2RenderingContext;
    };
    if (backend.device) {
      await backend.device.queue.onSubmittedWorkDone();
    } else if (backend.gl) {
      // Reading a pixel waits for the commands in progress.
      backend.gl.readPixels(
        0,
        0,
        1,
        1,
        backend.gl.RGBA,
        backend.gl.UNSIGNED_BYTE,
        new Uint8Array(4),
      );
    }
  }

  /**
   * Renders a frame. With `sceneryBehind`, the ground and the scenery are drawn first, then
   * the depth buffer is cleared and the track, the walls, the cars and the LiDAR are drawn
   * over them: whatever the camera, the scenery cannot hide them.
   */
  render(camera: THREE.Camera, sceneryBehind = false): void {
    const { renderer } = this;
    if (!sceneryBehind) {
      camera.layers.mask = ALL_LAYERS;
      renderer.render(this.scene, camera);
      return;
    }
    camera.layers.set(SCENERY_LAYER);
    renderer.render(this.scene, camera);
    // The second pass must keep the colours of the first: no clear, and no background, which
    // would paint over them.
    const { background } = this.scene;
    this.scene.background = null;
    renderer.autoClear = false;
    renderer.clearDepth();
    camera.layers.set(0);
    renderer.render(this.scene, camera);
    renderer.autoClear = true;
    this.scene.background = background;
    camera.layers.mask = ALL_LAYERS;
  }
}
