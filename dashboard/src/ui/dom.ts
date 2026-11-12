/** Small DOM helpers: the HUD is plain DOM, updated in place from the frame loop. */

type Child = Node | string | null | undefined | false;

type Attributes = Record<string, string | number | boolean | EventListener | undefined>;

/** Creates an element. `on*` attributes are event listeners, the others are attributes. */
export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attributes: Attributes = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (value === undefined || value === false) continue;
    if (typeof value === "function") {
      element.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === "class") {
      element.className = String(value);
    } else {
      element.setAttribute(key, value === true ? "" : String(value));
    }
  }
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    element.append(child);
  }
  return element;
}

/** Writes text only when it changed: most HUD values are the same from one frame to the next. */
export function setText(element: HTMLElement, text: string): void {
  if (element.textContent !== text) element.textContent = text;
}

/** Simulated time as `m:ss.d`. */
export function formatTime(seconds: number): string {
  const tenths = Math.floor(Math.max(seconds, 0) * 10 + 1e-6);
  const minutes = Math.floor(tenths / 600);
  const rest = tenths - minutes * 600;
  return `${minutes}:${String(Math.floor(rest / 10)).padStart(2, "0")}.${rest % 10}`;
}

const SVG_NS = "http://www.w3.org/2000/svg";

/** Inline icon from a path, drawn with the text colour. */
export function icon(path: string, size = 16): SVGSVGElement {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("aria-hidden", "true");
  const shape = document.createElementNS(SVG_NS, "path");
  shape.setAttribute("d", path);
  shape.setAttribute("fill", "currentColor");
  svg.append(shape);
  return svg;
}

export const ICONS = {
  play: "M4.5 2.6v10.8a.6.6 0 0 0 .92.5l8.4-5.4a.6.6 0 0 0 0-1L5.42 2.1a.6.6 0 0 0-.92.5Z",
  pause: "M4 2.5h2.6v11H4zM9.4 2.5H12v11H9.4z",
  restart:
    "M8 2.5a5.5 5.5 0 1 0 5.3 7h-1.6A4 4 0 1 1 8 4c1.1 0 2.1.45 2.8 1.2L8.5 7.5h5v-5l-1.6 1.6A5.5 5.5 0 0 0 8 2.5Z",
  folder:
    "M1.5 4a1.5 1.5 0 0 1 1.5-1.5h3.2l1.5 1.5H13A1.5 1.5 0 0 1 14.5 5.5v6A1.5 1.5 0 0 1 13 13H3a1.5 1.5 0 0 1-1.5-1.5V4Z",
  close:
    "M3.7 2.6 8 6.9l4.3-4.3 1.1 1.1L9.1 8l4.3 4.3-1.1 1.1L8 9.1l-4.3 4.3-1.1-1.1L6.9 8 2.6 3.7l1.1-1.1Z",
  frame: "M2 3.5h12v9H2v-9Zm1.5 1.5v6h9V5h-9Z",
} as const;
