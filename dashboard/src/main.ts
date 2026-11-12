import { App } from "./app";
import { h } from "./ui/dom";

declare global {
  interface Window {
    /** Debug access for the browser tests and the performance measurements. */
    dashboard?: Pick<App, "state" | "focus" | "benchmark">;
  }
}

const canvas = document.querySelector<HTMLCanvasElement>("#scene");
const hud = document.querySelector<HTMLElement>("#hud");
const markers = document.querySelector<HTMLElement>("#markers");
if (!canvas || !hud || !markers) throw new Error("the page markup is incomplete");

try {
  const app = await App.start(canvas, hud, markers);
  window.dashboard = app;
} catch (error) {
  console.error(error);
  hud.append(
    h(
      "div",
      { class: "library" },
      h(
        "div",
        { class: "panel library-card", role: "alert" },
        h("h1", {}, "La scène 3D ne peut pas démarrer"),
        h(
          "p",
          { class: "muted" },
          "Ce navigateur n'offre ni WebGPU ni WebGL 2. Activez l'accélération matérielle ou utilisez une version récente de Chrome ou de Firefox.",
        ),
        h("code", {}, error instanceof Error ? error.message : String(error)),
      ),
    ),
  );
}
