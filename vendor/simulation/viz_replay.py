"""
viz_replay.py — Offline top-down replay from a .npz episode file.

No sim dependency. Requires X11/display for GUI mode.

Usage:
    python viz_replay.py [--npz episode.npz] [--map examples/example_map.yaml] [--follow]

Controls:
    Play/Pause button — real-time playback (uses sim_dt)
    Left/Right arrow  — step by 1
    Shift+Left/Right  — step by 10
    Slider            — seek to any step
    Mouse wheel       — zoom in / zoom out centered on cursor
    Click + Drag      — pan / move map view
    Double-click / R  — reset zoom & pan to full track

Export:
    python viz_replay.py --npz episode.npz --record output.mp4 [--no-telemetry]
"""
import argparse
import glob
import math
import os
import sys
import time

import numpy as np
from PIL import Image
import yaml

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
RECORD_DIR = "/app/recordings"
_SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_maps_dir = os.path.join(_SIM_DIR, "maps", "examples")
_default_map = os.path.join(_maps_dir, "example_map.yaml") if os.path.exists(_maps_dir) else None

parser = argparse.ArgumentParser(description="Replay a recorded episode")
parser.add_argument("--npz", default=None, help="Episode NPZ (default: latest in /app/recordings/)")
parser.add_argument("--map", default=_default_map, help="Map YAML (auto-detected from episode if available)")
parser.add_argument("--follow", action="store_true", help="Re-center view on car 0 at each step")
parser.add_argument("--no-display", action="store_true",
                    help="Headless: save a PNG snapshot instead of opening a window")
parser.add_argument("--record", default=None, help="Export replay as MP4 video (ex: output.mp4)")
parser.add_argument("--telemetry", action="store_true", default=True,
                    help="Show telemetry overlay (default: on)")
parser.add_argument("--no-telemetry", dest="telemetry", action="store_false",
                    help="Hide telemetry overlay for cleaner video output")
parser.add_argument("--hold", type=float, default=5.0,
                    help="Seconds holding the final ranking card at the end of the MP4 (0 = off)")
args = parser.parse_args()

if args.npz is None:
    existing = sorted(glob.glob(os.path.join(RECORD_DIR, "episode_*.npz")))
    if not existing:
        sys.exit("No episode files in /app/recordings/. Use --npz <file>.")
    args.npz = existing[-1]
    print(f"Loading latest episode: {args.npz}")

# ---------------------------------------------------------------------------
# Constants & Pre-computed Geometry
# ---------------------------------------------------------------------------
_LIDAR_FOV   = 2.0 * math.pi
_CAR_HALF_L  = 0.29
_CAR_HALF_W  = 0.155
_CAR_CORNERS = np.array([
    [ _CAR_HALF_L,  _CAR_HALF_W],
    [ _CAR_HALF_L, -_CAR_HALF_W],
    [-_CAR_HALF_L, -_CAR_HALF_W],
    [-_CAR_HALF_L,  _CAR_HALF_W],
], dtype=np.float32)
_COLORS      = ["#FF6B35", "#4CC9F0", "#7BF178", "#F72585"]
_BG          = "#0D0D0D"

# ---------------------------------------------------------------------------
# Load episode
# ---------------------------------------------------------------------------
print(f"Loading {args.npz} ...")
ep = np.load(args.npz)
lidar        = ep["lidar"]       # (T, C, 100)
velocity     = ep["velocity"]    # (T, C)
steering     = ep["steering"]    # (T, C)
progress     = ep["progress"]    # (T, C)
lap_count    = ep["lap_count"]   # (T, C)
poses        = ep["poses"]       # (T, C, 3)  x y theta
actions      = ep["actions"]     # (T, C, 2)  spd steer
status       = ep["status"]      # (T, C)
ranks        = ep["ranks"] if "ranks" in ep else None
friction     = ep["friction"]    # (T,)
max_prog     = ep["max_prog"]    # (T, C)
compute_time = ep["compute_time"] if "compute_time" in ep else (ep["compute_times"] if "compute_times" in ep else None)
sim_dt       = float(ep["sim_dt"])
num_cars     = int(ep["num_cars"])
T            = lidar.shape[0]

calc_stats = {}
if compute_time is not None:
    for cid in range(num_cars):
        active_steps = np.where((status[:, cid] == 1) & (np.arange(T) > 0))[0]
        if len(active_steps) > 0:
            times_ms = compute_time[active_steps, cid] * 1000.0
            calc_stats[cid] = (float(np.mean(times_ms)), float(np.std(times_ms)))
        else:
            calc_stats[cid] = (0.0, 0.0)

if "team_names" in ep:
    team_names = [str(name).strip() for name in ep["team_names"]]
else:
    team_names = [f"Car_{i}" for i in range(num_cars)]

print(f"  {T} steps, {num_cars} car(s) ({', '.join(team_names)}), sim_dt={sim_dt}s")

# ---------------------------------------------------------------------------
# Resolve map
# ---------------------------------------------------------------------------
map_yaml = args.map
if "map_name" in ep:
    name = str(ep["map_name"]).strip()
    if name == "example":
        auto = os.path.join(_maps_dir, "example_map.yaml")
    else:
        auto = os.path.join(_SIM_DIR, "maps", name, f"{name}_map.yaml")
    if os.path.exists(auto):
        map_yaml = auto
        print(f"Auto-detected map: {auto}")

if map_yaml is None:
    sys.exit("No map found. Use --map <path>.")

with open(map_yaml) as f:
    meta = yaml.safe_load(f)
res = float(meta["resolution"])
ox, oy = float(meta["origin"][0]), float(meta["origin"][1])

img = np.array(Image.open(map_yaml.replace(".yaml", ".png")))
if img.ndim == 3:
    img = img[..., 0]
h, w = img.shape[:2]
extent = [ox, ox + w * res, oy, oy + h * res]

vals, counts = np.unique(img, return_counts=True)
bg_val = vals[np.argmax(counts)]
track_y, track_x = np.where(img != bg_val)
if len(track_x) == 0:
    track_y, track_x = np.where(img > 0)

if len(track_x) > 0:
    pad_m = 3.0
    track_xmin = max(extent[0], ox + track_x.min() * res - pad_m)
    track_xmax = min(extent[1], ox + track_x.max() * res + pad_m)
    track_ymin = max(extent[2], oy + (h - 1 - track_y.max()) * res - pad_m)
    track_ymax = min(extent[3], oy + (h - 1 - track_y.min()) * res + pad_m)
    track_extent = [track_xmin, track_xmax, track_ymin, track_ymax]
else:
    track_extent = extent[:]

half_win = max(track_extent[1] - track_extent[0], track_extent[3] - track_extent[2]) / 4.0

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
_n_lidar_beams = lidar.shape[2]
_LIDAR_REL_ANGLES = -_LIDAR_FOV / 2.0 + np.arange(_n_lidar_beams) * (_LIDAR_FOV / max(_n_lidar_beams - 1, 1))

def _car_vertices(x, y, c, s):
    return np.column_stack([
        _CAR_CORNERS[:, 0] * c - _CAR_CORNERS[:, 1] * s + x,
        _CAR_CORNERS[:, 0] * s + _CAR_CORNERS[:, 1] * c + y,
    ])

def _lidar_segments(x, y, theta, scan):
    angles = theta + _LIDAR_REL_ANGLES
    seg = np.empty((len(scan), 2, 2))
    seg[:, 0, 0] = x
    seg[:, 0, 1] = y
    seg[:, 1, 0] = x + scan * np.cos(angles)
    seg[:, 1, 1] = y + scan * np.sin(angles)
    return seg

# ---------------------------------------------------------------------------
# High-Speed OpenCV Video Pipeline (Bypasses Matplotlib completely on --record)
# ---------------------------------------------------------------------------
OUT_W, OUT_H = 1920, 1080
MAX_SS = 3          # max map supersampling (quality vs ms/frame)

_BODY = np.array([[.29,.07],[.24,.135],[-.19,.155],[-.27,.12],[-.29,0.],
                  [-.27,-.12],[-.19,-.155],[.24,-.135],[.29,-.07]], np.float32)
_COCKPIT = np.array([[.06,.085],[-.02,.10],[-.15,.09],
                     [-.15,-.09],[-.02,-.10],[.06,-.085]], np.float32)
_WHEEL = np.array([[.075,.028],[.075,-.028],[-.075,-.028],[-.075,.028]], np.float32)
_WHEEL_POS = [(.19,.147),(.19,-.147),(-.19,.147),(-.19,-.147)]
_BGR_COLORS = [(53,107,255),(240,201,76),(120,241,123),(133,37,247)]


def _order_at(i):
    """Standings at frame i: ranks if available, else FINISHED > active > DNF."""
    if ranks is not None:
        return sorted(range(num_cars), key=lambda c: (ranks[i,c], -lap_count[i,c], -progress[i,c]))
    return sorted(range(num_cars),
                  key=lambda c: (0 if status[i,c] == 2 else (1 if status[i,c] == 1 else 2),
                                 -lap_count[i,c], -progress[i,c]))


def _export_video_opencv(output_path, follow=False, show_telemetry=True, hold_sec=5.0):
    import cv2
    t0 = time.perf_counter()
    fps = max(1, round(1.0 / sim_dt))

    # --- supersampling factor from the actual viewport size ---
    if follow:
        vw = vh = 2.0 * half_win
    else:
        vw, vh = track_extent[1]-track_extent[0], track_extent[3]-track_extent[2]
    #SS = int(np.clip(math.ceil(max(OUT_W*res/vw, OUT_H*res/vh)), 1, MAX_SS))
    SS = 1
    lw = max(1, SS)

    # --- base map: single upscale + light edge softening ---
    if img.ndim == 2:
        base = cv2.cvtColor(((1.0 - img.astype(np.float32)/255.0)*215 + 22).astype(np.uint8),
                            cv2.COLOR_GRAY2BGR)
    else:
        base = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    base = cv2.resize(base, (w*SS, h*SS), interpolation=cv2.INTER_LINEAR)
    if SS > 1:
        base = cv2.GaussianBlur(base, (0, 0), SS*0.45)   # removes the wall-pixel staircase

    def w2p(x, y):  # world -> pixel in the supersampled image
        return np.int32((x - ox)/res*SS), np.int32((h - 1 - (y - oy)/res)*SS)

    u0_t, v1_t = w2p(track_extent[0], track_extent[2])
    u1_t, v0_t = w2p(track_extent[1], track_extent[3])
    u0_t, u1_t = max(0, int(u0_t)), min(w*SS, int(u1_t))
    v0_t, v1_t = max(0, int(v0_t)), min(h*SS, int(v1_t))
    half_px = int(half_win/res*SS)

    # --- static trajectory (single car), halo + sharp line ---
    if num_cars == 1:
        act = np.where((status[:,0] == 1) & (poses[:,0,0] < 5000.0))[0]
        if len(act) > 1:
            u, v = w2p(poses[act,0,0], poses[act,0,1])
            pts = [np.stack([u, v], 1)]
            cv2.polylines(base, pts, False, (180,205,255), lw*3, cv2.LINE_AA)
            cv2.polylines(base, pts, False, (53,107,255),  lw*2, cv2.LINE_AA)

    def _blend(dst, draw, alpha, pts):
        """Blend a stroke over the points' bbox only (avoids a full-frame copy)."""
        x, y, bw, bh = cv2.boundingRect(pts)
        p = lw*3
        x, y = max(0, x-p), max(0, y-p)
        bw, bh = min(dst.shape[1]-x, bw+2*p), min(dst.shape[0]-y, bh+2*p)
        if bw <= 0 or bh <= 0:
            return
        roi = dst[y:y+bh, x:x+bw]
        ov = roi.copy()
        draw(ov, x, y)
        cv2.addWeighted(ov, alpha, roi, 1.0-alpha, 0, roi)

    def _draw_car(dst, x, y, th, col, ou, ov_, label=None):
        c, s = math.cos(th), math.sin(th)
        def P(p, dx=0.0, dy=0.0):
            u, v = w2p(p[:,0]*c - p[:,1]*s + x + dx, p[:,0]*s + p[:,1]*c + y + dy)
            return np.stack([u-ou, v-ov_], 1).astype(np.int32)
        body = P(_BODY)
        dark  = tuple(int(q*0.38) for q in col)
        light = tuple(min(255, int(q*0.45 + 140)) for q in col)

        for wx, wy in _WHEEL_POS:
            cv2.fillPoly(dst, [P(_WHEEL + (wx, wy))], (28,28,28), cv2.LINE_AA)
        cv2.fillPoly(dst, [body], col, cv2.LINE_AA)
        cv2.polylines(dst, [body], True, dark, lw, cv2.LINE_AA)
        cv2.fillPoly(dst, [P(_COCKPIT)], dark, cv2.LINE_AA)
        # light edge on the left flank = depth
        cv2.polylines(dst, [P(_BODY[:4])], False, light, max(1, lw), cv2.LINE_AA)
        for hy in (.085, -.085):   # headlights: heading readable at a glance
            hu, hv = w2p(x + .265*c - hy*s, y + .265*s + hy*c)
            cv2.circle(dst, (int(hu)-ou, int(hv)-ov_), lw, (235,245,255), -1, cv2.LINE_AA)

        if label:
            tu, tv = w2p(x, y + .55)
            f = 0.45*SS
            (tw_, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, f, lw)
            org = (int(tu)-ou-tw_//2, int(tv)-ov_)
            cv2.putText(dst, label, org, cv2.FONT_HERSHEY_DUPLEX, f, (255,255,255), lw*1, cv2.LINE_AA)
            cv2.putText(dst, label, org, cv2.FONT_HERSHEY_DUPLEX, f, col, lw, cv2.LINE_AA)

    # --- Writer : pipe ffmpeg/libx264 si dispo, sinon cv2 (mp4v, qualite degradee) ---
    def _open_writer(path, fps, size):
        import shutil, subprocess
        if shutil.which("ffmpeg"):
            p = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-",
                "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
                stdin=subprocess.PIPE)
            return (lambda f: p.stdin.write(f.tobytes()),
                    lambda: (p.stdin.close(), p.wait()), "libx264 crf18")
        wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        print("WARN: ffmpeg introuvable -> mp4v (qualite degradee a 1080p).")
        return wr.write, wr.release, "mp4v"

    write_frame, close_writer, codec = _open_writer(output_path, fps, (OUT_W, OUT_H))

    for i in range(T):
        if follow and num_cars > 0:
            cu, cv_ = w2p(poses[i,0,0], poses[i,0,1])
            u0, u1 = max(0, int(cu)-half_px), min(w*SS, int(cu)+half_px)
            v0, v1 = max(0, int(cv_)-half_px), min(h*SS, int(cv_)+half_px)
        else:
            u0, u1, v0, v1 = u0_t, u1_t, v0_t, v1_t
        frame = base[v0:v1, u0:u1].copy()

        # LiDAR: filled visibility polygon + outline (1 fillPoly, not 100 lines)
        if num_cars == 1 and status[i,0] == 1 and poses[i,0,0] < 5000.0:
            x0, y0, th0 = poses[i,0]
            ang = th0 + _LIDAR_REL_ANGLES
            bu, bv = w2p(x0 + lidar[i,0]*np.cos(ang), y0 + lidar[i,0]*np.sin(ang))
            fan = np.stack([bu-u0, bv-v0], 1).astype(np.int32)
            _blend(frame, lambda o, dx, dy: cv2.fillPoly(
                o, [fan - (dx, dy)], (120,235,90), cv2.LINE_AA), 0.22, fan)
            cv2.polylines(frame, [fan], True, (95,205,60), max(1, lw), cv2.LINE_AA)

        for cid in range(num_cars):
            x, y, th = poses[i, cid]
            if status[i, cid] == 1 and x < 5000.0:
                _draw_car(frame, x, y, th, _BGR_COLORS[cid % len(_BGR_COLORS)], u0, v0,
                          label=team_names[cid] if num_cars > 1 else None)

        # letterbox: keeps aspect (a direct resize distorted the track)
        sc = min(OUT_W/frame.shape[1], OUT_H/frame.shape[0])
        nw, nh = max(1, int(frame.shape[1]*sc)), max(1, int(frame.shape[0]*sc))
        canvas = np.full((OUT_H, OUT_W, 3), 13, np.uint8)
        xo, yo = (OUT_W-nw)//2, (OUT_H-nh)//2
        canvas[yo:yo+nh, xo:xo+nw] = cv2.resize(
            frame, (nw, nh), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)

        if show_telemetry:
            hs = OUT_H / 720.0
            lines = [f"Step {i:<4d}  |  t = {i*sim_dt:.2f}s", ""]
            if num_cars > 1:
                lines.append("-- CLASSEMENT --")
                for p, c in enumerate(_order_at(i), 1):
                    st = {1:"active", 2:"FINISHED"}.get(status[i,c], "DNF")
                    lines.append(f" {p} - {team_names[c]:<8s} L{lap_count[i,c]} ({st})")
                lines.append("")
            for cid in range(num_cars):
                if cid > 0:
                    lines.append("")
                st = {1:"active", 2:"FINISHED"}.get(status[i,cid], "DNF")
                lines += [f"--- {team_names[cid]} ({st}) ---",
                          f"vel: {velocity[i,cid]:+.2f}m/s  steer: {steering[i,cid]:+.3f}",
                          f"cmd: {actions[i,cid,0]:+.2f}v | {actions[i,cid,1]:+.3f}rad",
                          f"prog: {progress[i,cid]:.2f} (max {max_prog[i,cid]:.2f})  lap: {lap_count[i,cid]}"]
            lines.append("")
            lines.append(f"friction: {friction[i]:.3f}")

            lh, hud_w = int(16*hs), int(345*hs)
            hud_h = min(OUT_H - int(20*hs), int(15*hs) + len(lines)*lh)
            m = int(12*hs)
            ov = canvas.copy()
            cv2.rectangle(ov, (m, m), (m+hud_w, m+hud_h), (13,13,13), -1)
            cv2.addWeighted(ov, 0.80, canvas, 0.20, 0, canvas)
            cv2.rectangle(canvas, (m, m), (m+hud_w, m+hud_h), (70,70,70), max(1, int(hs)))
            yt = m + int(16*hs)
            for ln in lines:
                cv2.putText(canvas, ln, (m+int(10*hs), yt), cv2.FONT_HERSHEY_SIMPLEX,
                            0.38*hs, (238,238,238), max(1, int(hs)), cv2.LINE_AA)
                yt += lh

        write_frame(canvas)
        if (i+1) % 200 == 0 or i == T-1:
            print(f"Export video : {i+1}/{T} frames ({100*(i+1)//T}%)")

    # --- Carte de fin : le classement final tenu quelques secondes ---
    # A l'arrivee toutes les voitures sont teleportees hors carte (_finish/_freeze_dnf),
    # donc geler la derniere frame telle quelle montrerait une piste vide.
    n_hold = int(round(hold_sec * fps)) if (hold_sec > 0 and num_cars > 0) else 0
    if n_hold:
        hs = OUT_H / 720.0
        rows = [(p, team_names[c], lap_count[T-1,c],
                 {1:"active", 2:"FINISHED"}.get(status[T-1,c], "DNF"))
                for p, c in enumerate(_order_at(T-1), 1)]
        card = cv2.addWeighted(canvas, 0.28, np.zeros_like(canvas), 0.72, 0)
        pw, ph = int(520*hs), int((86 + 44*len(rows))*hs)
        x0, y0 = (OUT_W-pw)//2, (OUT_H-ph)//2
        cv2.rectangle(card, (x0,y0), (x0+pw,y0+ph), (13,13,13), -1)
        cv2.rectangle(card, (x0,y0), (x0+pw,y0+ph), (95,205,60), max(1, int(2*hs)))
        cv2.putText(card, "CLASSEMENT FINAL", (x0+int(26*hs), y0+int(50*hs)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.90*hs, (95,205,60), max(1, int(2*hs)), cv2.LINE_AA)
        for k, (p, nm, lp, st) in enumerate(rows):
            cv2.putText(card, f"{p}.  {nm:<10s} L{lp}  ({st})",
                        (x0+int(26*hs), y0+int((88 + 44*k)*hs)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.60*hs, (120,120,120) if st == "DNF" else (238,238,238),
                        max(1, int(hs)), cv2.LINE_AA)
        for _ in range(n_hold):
            write_frame(card)

    close_writer()
    el = time.perf_counter() - t0
    nf = T + n_hold
    print(f"Saved -> {output_path} ({nf} frames @{OUT_W}x{OUT_H}, SS={SS}, {codec}"
          + (f", +{hold_sec:.1f}s carte de fin" if n_hold else "")
          + f") in {el:.2f}s ({nf/el:.1f} FPS)")

if args.record:
    _export_video_opencv(args.record, follow=args.follow, show_telemetry=args.telemetry,
                         hold_sec=args.hold)
    sys.exit(0)

# ---------------------------------------------------------------------------
# Centerline
# ---------------------------------------------------------------------------
def _load_centerline():
    """(N+1, 2) boucle fermee de la ligne de reference du circuit, ou None."""
    name = str(ep["map_name"]).strip() if "map_name" in ep else None
    cands = []
    if name:
        cands.append((os.path.join(_SIM_DIR, "maps", name, f"{name}_centerline.csv"), ",", 0, 0, 1))
        cands.append((os.path.join(_maps_dir, f"{name}_centerline.csv"), ",", 0, 0, 1))
    cands.append((os.path.join(_maps_dir, "example_waypoints.csv"), ";", 3, 1, 2))
    for path, delim, skip, xi, yi in cands:
        if not os.path.exists(path):
            continue
        try:
            d = np.loadtxt(path, delimiter=delim, skiprows=skip, comments="#")
        except (IOError, ValueError):
            continue
        if d.ndim != 2 or len(d) < 2:
            continue
        pts = d[:, [xi, yi]]
        return np.vstack([pts, pts[:1]])
    return None


# ---------------------------------------------------------------------------
# Matplotlib Setup (GUI and Headless Snapshots only)
# ---------------------------------------------------------------------------
import matplotlib
_HEADLESS = args.no_display or not os.environ.get("DISPLAY")
matplotlib.use("Agg" if _HEADLESS else "TkAgg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider, Button

fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor(_BG)

ax_map = fig.add_axes([0.01, 0.08, 0.98, 0.90])
ax_tel = fig.add_axes([0.01, 0.08, 0.24, 0.90])
ax_sld = fig.add_axes([0.01, 0.03, 0.79, 0.035])
ax_btn = fig.add_axes([0.83, 0.025, 0.14, 0.045])

for ax in (ax_map, ax_tel):
    ax.set_facecolor(_BG)
    ax.axis("off")

ax_map.set_xlim(track_extent[0], track_extent[1])
ax_map.set_ylim(track_extent[2], track_extent[3])
ax_map.set_aspect("equal")
ax_map.imshow(np.flipud(img), extent=extent, origin="lower",
              cmap="gray_r", alpha=0.85, zorder=1, interpolation="nearest")

# Centerline du circuit (affichage seulement : --record sort avant ce bloc)
_cl = _load_centerline()
if _cl is not None:
    ax_map.plot(_cl[:, 0], _cl[:, 1], color="#0A2463", lw=0.4, alpha=0.95, zorder=2)

if num_cars == 1:
    for cid in range(num_cars):
        color = _COLORS[cid % len(_COLORS)]
        active_idx = np.where((status[:, cid] == 1) & (poses[:, cid, 0] < 5000.0))[0]
        if len(active_idx) > 1:
            ax_map.plot(poses[active_idx, cid, 0], poses[active_idx, cid, 1],
                        color=color, lw=2.0, alpha=0.5, zorder=2)

car_polys, car_arrows, traj_markers = [], [], []
for cid in range(num_cars):
    color = _COLORS[cid % len(_COLORS)]
    poly = Polygon(np.zeros((4, 2)), closed=True, fc=color, ec="white",
                   lw=0.8, alpha=0.9, zorder=4)
    ax_map.add_patch(poly)
    arrow, = ax_map.plot([], [], color="white", lw=1.2, zorder=5, solid_capstyle="butt")
    mk, = ax_map.plot([], [], "o", color=color, ms=3, zorder=6,
                      markeredgecolor="white", markeredgewidth=0.8)
    car_polys.append(poly)
    car_arrows.append(arrow)
    traj_markers.append(mk)

lidar_lc = LineCollection([], color="#00BB55", lw=1.5, alpha=0.7, zorder=3)
ax_map.add_collection(lidar_lc)

if not args.telemetry:
    ax_tel.set_visible(False)

tel_fontsize = 12 if num_cars == 1 else (10 if num_cars == 2 else (8.5 if num_cars == 3 else 7.5))
tel_text = ax_tel.text(0.02, 0.99, "", transform=ax_tel.transAxes, va="top", ha="left",
                       fontsize=tel_fontsize, color="white", family="monospace",
                       bbox=dict(boxstyle="round,pad=0.35", fc="#0D0D0D", ec="#333333", alpha=0.80))

slider = Slider(ax_sld, "Step", 0, T - 1, valinit=0, valstep=1,
                color="#1A6FD4", track_color="#CCCCCC")
ax_sld.xaxis.label.set_color("white")
ax_sld.tick_params(colors="white")

button = Button(ax_btn, "\u25b6  Play", color="#444444", hovercolor="#555555")
button.label.set_color("white")
button.label.set_fontweight("bold")
_playing = False

def _update(step_i):
    step_i = int(np.clip(step_i, 0, T - 1))

    for cid in range(num_cars):
        x, y, theta = poses[step_i, cid]
        if status[step_i, cid] == 1 and x < 5000.0:
            c, s = math.cos(theta), math.sin(theta)
            car_polys[cid].set_xy(_car_vertices(x, y, c, s))
            car_polys[cid].set_visible(True)
            car_arrows[cid].set_data([x, x + 0.45 * c], [y, y + 0.45 * s])
            car_arrows[cid].set_visible(True)
            traj_markers[cid].set_data([x], [y])
        else:
            car_polys[cid].set_visible(False)
            car_arrows[cid].set_visible(False)
            traj_markers[cid].set_data([], [])

    if num_cars == 1 and status[step_i, 0] == 1 and poses[step_i, 0, 0] < 5000.0:
        x0, y0, th0 = poses[step_i, 0]
        lidar_lc.set_segments(_lidar_segments(x0, y0, th0, lidar[step_i, 0]))
    else:
        lidar_lc.set_segments([])

    if args.follow and num_cars > 0:
        x0, y0 = poses[step_i, 0][:2]
        ax_map.set_xlim(x0 - half_win, x0 + half_win)
        ax_map.set_ylim(y0 - half_win, y0 + half_win)

    lines = [f"Step {step_i:<4d}  |  t = {step_i * sim_dt:.2f}s", ""]

    if num_cars > 1:
        lines.append("── CLASSEMENT ──")
        if ranks is not None:
            order = sorted(range(num_cars), key=lambda c: (ranks[step_i, c], -lap_count[step_i, c], -progress[step_i, c]))
        else:
            order = sorted(
                range(num_cars),
                key=lambda c: (
                    0 if status[step_i, c] == 2 else (1 if status[step_i, c] == 1 else 2),
                    -lap_count[step_i, c],
                    -progress[step_i, c]
                )
            )

        for p_idx, c in enumerate(order, start=1):
            st_code = status[step_i, c]
            st = "active" if st_code == 1 else ("FINISHED" if st_code == 2 else "DNF")
            tname = team_names[c]
            lines.append(f" {p_idx} - {tname:<10s} L{lap_count[step_i, c]} ({st})")
        lines.append("")
    for cid in range(num_cars):
        st_code = status[step_i, cid]
        if st_code == 1:
            st = "active"
        elif st_code == 2:
            st = "FINISHED"
        else:
            st = "DNF"
        tname = team_names[cid]
        lines += [
            f"─── {tname} ({st}) ───", "",
            f"velocity : {velocity[step_i, cid]:+.2f} m/s",
            f"steering : {steering[step_i, cid]:+.4f} rad",
            f"cmd speed: {actions[step_i, cid, 0]:+.2f}",
            f"cmd steer: {actions[step_i, cid, 1]:+.4f}", "",
            f"progress : {progress[step_i, cid]:.3f}",
            f"max prog : {max_prog[step_i, cid]:.3f}",
            f"lap      : {lap_count[step_i, cid]}", "",
        ]
        if compute_time is not None:
            cur_ms = compute_time[step_i, cid] * 1000.0
            mean_ms, std_ms = calc_stats.get(cid, (0.0, 0.0))
            lines += [
                f"calc time: {cur_ms:4.2f} ms ({mean_ms:.2f}±{std_ms:.2f}ms)", "",
            ]
    lines.append(f"friction : {friction[step_i]:.3f}")
    tel_text.set_text("\n".join(lines))
    fig.canvas.draw_idle()

slider.on_changed(_update)

def _toggle(event=None):
    global _playing
    _playing = not _playing
    if _playing:
        if int(slider.val) >= T - 1:
            slider.set_val(0)
        button.label.set_text("\u23f8  Pause")
        button.ax.set_facecolor("#1A6FD4")
    else:
        button.label.set_text("\u25b6  Play")
        button.ax.set_facecolor("#444444")
    fig.canvas.draw_idle()

button.on_clicked(_toggle)

_pan_start = None

def _reset_view():
    ax_map.set_xlim(track_extent[0], track_extent[1])
    ax_map.set_ylim(track_extent[2], track_extent[3])
    fig.canvas.draw_idle()

def _on_scroll(event):
    if event.inaxes != ax_map or args.follow:
        return
    scale = 0.82 if event.button == "up" else 1.22
    cur_xlim = ax_map.get_xlim()
    cur_ylim = ax_map.get_ylim()
    xdata = event.xdata if event.xdata is not None else (cur_xlim[0] + cur_xlim[1]) / 2.0
    ydata = event.ydata if event.ydata is not None else (cur_ylim[0] + cur_ylim[1]) / 2.0

    new_w = (cur_xlim[1] - cur_xlim[0]) * scale
    new_h = (cur_ylim[1] - cur_ylim[0]) * scale

    relx = (cur_xlim[1] - xdata) / max(cur_xlim[1] - cur_xlim[0], 1e-6)
    rely = (cur_ylim[1] - ydata) / max(cur_ylim[1] - cur_ylim[0], 1e-6)

    ax_map.set_xlim([xdata - new_w * (1 - relx), xdata + new_w * relx])
    ax_map.set_ylim([ydata - new_h * (1 - rely), ydata + new_h * rely])
    fig.canvas.draw_idle()

def _on_press(event):
    global _pan_start
    if event.inaxes == ax_map and not args.follow:
        if event.dblclick:
            _reset_view()
            return
        if event.button in (1, 2, 3):
            _pan_start = (event.x, event.y, ax_map.get_xlim(), ax_map.get_ylim())

def _on_release(event):
    global _pan_start
    _pan_start = None

def _on_motion(event):
    global _pan_start
    if _pan_start is None or event.inaxes != ax_map or args.follow:
        return
    x0, y0, (xlim0, xlim1), (ylim0, ylim1) = _pan_start
    dx = event.x - x0
    dy = event.y - y0
    bbox = ax_map.get_window_extent()
    if bbox.width <= 0 or bbox.height <= 0:
        return
    scale_x = (xlim1 - xlim0) / bbox.width
    scale_y = (ylim1 - ylim0) / bbox.height
    ax_map.set_xlim([xlim0 - dx * scale_x, xlim1 - dx * scale_x])
    ax_map.set_ylim([ylim0 - dy * scale_y, ylim1 - dy * scale_y])
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("scroll_event", _on_scroll)
fig.canvas.mpl_connect("button_press_event", _on_press)
fig.canvas.mpl_connect("button_release_event", _on_release)
fig.canvas.mpl_connect("motion_notify_event", _on_motion)

def _on_key(event):
    global _playing
    if event.key in ("r", "R") and not args.follow:
        _reset_view()
        return

    step = {"left": -1, "right": 1, "shift+left": -10, "shift+right": 10}.get(event.key)
    if step is None:
        return
    if _playing:
        _toggle()
    slider.set_val(int(np.clip(slider.val + step, 0, T - 1)))

fig.canvas.mpl_connect("key_press_event", _on_key)

# ---------------------------------------------------------------------------
# Main loop (Interactive GUI / Headless snapshot)
# ---------------------------------------------------------------------------
_update(0)

if _HEADLESS:
    snap_path = args.npz.replace(".npz", "_snapshot.png")
    fig.savefig(snap_path, dpi=150)
    print(f"Headless mode: snapshot saved to {snap_path}")
    sys.exit(0)

plt.show(block=False)

_last_step_time = time.perf_counter()
while plt.fignum_exists(fig.number):
    if _playing:
        now = time.perf_counter()
        if now - _last_step_time >= sim_dt:
            _last_step_time = now
            v = int(slider.val)
            if v >= T - 1:
                _toggle()
            else:
                slider.set_val(v + 1)
    else:
        _last_step_time = time.perf_counter()
    fig.canvas.flush_events()
    #time.sleep(0.005)