"""Operator-console deck figures: crops of the REAL console replaying logged EVAL runs + an annotation layer.

What this does
--------------
1. Serves the repo root on 127.0.0.1 (free port) and opens operator_ui/index.html in Playwright's bundled
   headless Chromium at a 1600 x 912 CSS-px viewport (device scale 2). At >= 1600 px wide the console uses
   its own large-screen tokens (15 px text). A FULL run's mission.json + autonomy/telemetry.jsonl are loaded
   through the console's own ?deterministic=1&mission=&replay= parameters, the console's zoom button is
   clicked and window.consoleSeek(t) shows the state at the chosen mission time. Nothing in operator_ui/ is
   edited, restyled or injected: the pixels inside the crops are the console's own rendering.
2. Crops three console regions: the mode banner (its empty middle is removed so the mode/reason and the
   IN MODE / DIST TO B readouts sit in one row), the ego-costmap (the 16 m costmap square plus a small
   margin, under the panel's own header) and the speed-governor panel (header + speed and R_cert bars; its
   squeezed sparkline is left out). The integrity gauge, downlink, command buttons, top bar and event log are
   left out on purpose: in tier-0 the integrity gauge is a fixed placeholder (q = 1.00, node.py sets
   q, p_fail = 1.0, 0.0 when there are no images), the link figures are computed from logged packet sizes
   (no radio), and the buttons are not wired to the autonomy (docs/PIPELINE_STATUS.md).
3. Composes one figure per moment with matplotlib (deck palette): the crops placed at 1.858 output px per
   CSS px (so the console's 10.8 px tick labels land at >= 20 px in the ~1800 px wide image), numbered
   markers on the costmap at cell positions computed from the decoded 4-bit costmap in the same telemetry
   packet, a speed / v_cap chart of the whole run (the same telemetry packets the console's sparkline
   plots), short notes, an honest colour key and a provenance line.

Every number drawn is either read back from the console DOM, taken from the logged telemetry packet /
per-tick debug record / result.json / autonomy.log of that run, or computed here from those (stated in
deck_assets/final/console.json). All runs are tier-0 simulation (synthetic depth sensor, no camera images).

Run from the repo root:  python deck_assets/final/src/console_shots.py
Outputs: deck_assets/final/console_<k>.png / .svg and deck_assets/final/console.json
"""
from __future__ import annotations

import io
import json
import math
import re
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402
from scipy import ndimage  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "deck_assets" / "final"
RUNS = "results/runs_eval_tier0/FULL"

# ---------------------------------------------------------------- deck palette (matches the other deck figures)
NAVY, GREY_BASE, GREEN, RED, AMBER, WATER = "#1F497D", "#9A9A9A", "#2E7D32", "#C62828", "#D98E04", "#1E88E5"
INK, GRID, WHITE, MUTED = "#1A1A1A", "#E6E6E6", "#FFFFFF", "#6E6E6E"
CONSOLE_BORDER = "#E2E8F0"  # operator_ui/style.css --border (frame drawn where a crop cuts a console panel)

# ---------------------------------------------------------------- console geometry / render settings
VIEWPORT = {"width": 1600, "height": 912}  # >= 1600 px: the console's own large-screen tokens (--fs 15 px)
DSF = 2.0  # screenshot device scale factor
S = 1.858  # output px per CSS px in the composed figure (10.8 px console ticks -> 20.1 px)
COSTMAP_N, COSTMAP_RES_M = 64, 0.25  # operator_ui/app.js COSTMAP_N / COSTMAP_RES (16 m x 16 m ego costmap)
MAP_CROP_W_CSS = 470.0  # costmap crop width [CSS px]: the 439.5 px costmap square + ~15 px margin per side

# ---------------------------------------------------------------- figure layout [output px]
FIG_W, DPI = 1800, 200
M_L, GAP_COL = 24, 30
COL_L_W = round(MAP_CROP_W_CSS * S)  # 873
COL_R_W = round(460 * S)  # 855 (console --side-w 460 px)
X_R = M_L + COL_L_W + GAP_COL
BANNER_W = X_R + COL_R_W - M_L

# console cell colours (operator_ui/app.js CELL_RGB == metagross.contracts.messages.CELL_COLORS)
CELL_RGB = {"UNSEEN": (203, 208, 214), "GROUND": (34, 160, 90), "OLIVE": (110, 120, 40), "LETHAL": (220, 50, 47),
            "DITCH_CANDIDATE": (200, 40, 160), "CREST_SHADOW": (240, 160, 30), "OCCLUDED": (150, 156, 164),
            "WATER": (20, 140, 190), "DYNAMIC": (250, 90, 20)}
CONSOLE_PLAN = "#1D4ED8"  # app.js THEME.accent (planned path)
CONSOLE_TRAIL = "#2563EB"  # app.js THEME.vo (trail)
# 4-bit codes (metagross/autonomy/link/codec.py U4_*; contracts COSTMAP_U4_CODES)
U4 = {"UNSEEN": 0, "OCCLUDED": 9, "CREST_SHADOW": 10, "WATER": 11, "DITCH_CANDIDATE": 12, "DYNAMIC": 13,
      "DITCH": 14, "LETHAL": 15}
U4_NAME = {v: k for k, v in U4.items()}

# governor constants for the stopping-distance estimate (metagross/config/defaults.py; a and T_r are also
# logged per tick in the debug record and checked against these below)
GOVERNOR_MARGIN_M = 0.5

FAMILY_TXT = {"F1_trail": "F1 trail", "F3_crest_ditch": "F3 crest + ditch"}

# (k, EVAL seed, seek time [s], console zoom extent [m])
MOMENTS = [(1, 44, 22.8, 16), (2, 48, 6.6, 16), (3, 48, 26.4, 16)]

DOM_FIELDS = {"mission": "mission-id", "mode": "mode-chip", "reason": "mode-reason", "in_mode": "mode-since",
              "dist_to_b": "dist-goal", "clock": "clock-mission", "speed_vcap": "speed-val", "r_cert": "rcert-val",
              "q": "q-text", "p_fail": "pfail-val", "pos_sigma": "sigma-val", "vo": "vo-val", "slip": "slip-val",
              "link_kbps": "link-kbps", "link_loss": "link-loss", "packets": "pk-n", "pk_mean": "pk-mean"}
BOX_SELECTORS = {"banner": "#banner", "banner_right0": "#banner .kv.right", "map_head": "#map-panel .panel-h",
                 "map": "#map", "gov": "#side .panel:first-child", "rcert_ticks": "#rcert-ticks"}


# ================================================================ run data
def run_dir(seed: int) -> Path:
    return REPO / RUNS / f"{seed:03d}"


def telemetry(seed: int) -> list[dict]:
    return [json.loads(l) for l in (run_dir(seed) / "autonomy" / "telemetry.jsonl").read_text().splitlines() if l.strip()]


def packet_at(rows: list[dict], t: float) -> tuple[dict, int]:
    """Last packet with t_pkt <= t (what consoleSeek(t) ends on) and the number of packets applied."""
    shown = [r for r in rows if r["t"] <= t + 1e-9]
    return shown[-1], len(shown)


def decode_costmap(hex_str: str) -> np.ndarray:
    """costmap_u4_hex -> (64, 64) codes, high nibble first (as app.js decodeCostmap / codec.pack_u4)."""
    b = np.frombuffer(bytes.fromhex(hex_str), np.uint8)
    out = np.empty(b.size * 2, np.uint8)
    out[0::2], out[1::2] = b >> 4, b & 15
    return out.reshape(COSTMAP_N, COSTMAP_N)


def cell_body_xy() -> tuple[np.ndarray, np.ndarray]:
    """Body-frame centre (x fwd, y left) [m] of every costmap cell: row 0 = +8 m forward, col 0 = +8 m left."""
    r, c = np.mgrid[0:COSTMAP_N, 0:COSTMAP_N]
    half = COSTMAP_N * COSTMAP_RES_M / 2
    return half - (r + 0.5) * COSTMAP_RES_M, half - (c + 0.5) * COSTMAP_RES_M


def to_body(pose: list[float], x: float, y: float) -> tuple[float, float]:
    """A-frame point -> body frame of pose (same as app.js toBody)."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    dx, dy = x - pose[0], y - pose[1]
    return c * dx + s * dy, -s * dx + c * dy


def debug_tick(seed: int, t: float) -> dict:
    """Per-tick debug record (5 Hz) at mission time t: governor terms, a, T_r, measured speed, command."""
    tick = int(round(t / 0.2))
    z = np.load(run_dir(seed) / "autonomy" / "debug" / f"tick_{tick:06d}.npz")
    ex = json.loads(str(z["extras_json"]))
    assert abs(float(z["t"]) - t) < 1e-6, (seed, t, float(z["t"]))
    return {"tick": tick, "t": float(z["t"]), "v_cap_mps": float(z["v_cap_mps"]), "r_cert_m": float(z["r_cert_m"]),
            "reason": ex["reason"], "gov_terms": ex["gov_terms"], "gov_binding": ex["gov_binding"], "a_mps2": ex["a_mps2"],
            "t_r_s": ex["t_r_s"], "r_path_m": ex["r_path_m"], "cmd_v": ex["cmd_v"], "speed_meas": ex["speed_meas"],
            "r_vis_m": json.loads(str(z["health_json"]))["r_vis_m"]}


def supervisor_transitions(seed: int) -> list[dict]:
    """Mode changes from autonomy.log, e.g. 't=6.40 mode NOMINAL -> STOP_AND_LOOK (NO_PROGRESS look)'."""
    pat = re.compile(r"t=([0-9.]+) mode (\w+) -> (\w+) \((.*)\)")
    out = []
    for line in (run_dir(seed) / "autonomy" / "autonomy.log").read_text().splitlines():
        m = pat.search(line)
        if m:
            out.append({"t": float(m.group(1)), "from": m.group(2), "to": m.group(3), "why": m.group(4)})
    return out


def stop_distance(v: float, a: float, t_r: float, b: float = GOVERNOR_MARGIN_M) -> float:
    """Governor stopping distance d_stop(v) = v^2 / (2 a) + v T_r + B [m] (planning/governor.py docstring)."""
    return v * v / (2.0 * a) + v * t_r + b


def components(codes: np.ndarray, code_set: tuple[int, ...]) -> list[dict]:
    """Connected regions (4-connected) of the given codes, largest first, with body-frame stats."""
    xb, yb = cell_body_xy()
    mask = np.isin(codes, code_set)
    lab, n = ndimage.label(mask)
    comps = []
    for i in range(1, n + 1):
        m = lab == i
        comps.append({"mask": m, "n_cells": int(m.sum()), "centroid_body_m": [round(float(xb[m].mean()), 2), round(float(yb[m].mean()), 2)],
                      "nearest_m": round(float(np.hypot(xb[m], yb[m]).min()), 2)})
    return sorted(comps, key=lambda d: -d["n_cells"])


def anchor_cell(mask: np.ndarray, target_xy: tuple[float, float], avoid_xy: list[tuple[float, float]], min_gap_m: float) -> tuple[int, int]:
    """Marker cell for a region: well inside `mask` (distance-to-edge first), then nearest `target_xy` (body m),
    keeping >= min_gap_m from every point in `avoid_xy` (the rover and the planned path)."""
    xb, yb = cell_body_xy()
    ok = mask.copy()
    for ax, ay in avoid_xy:
        ok &= np.hypot(xb - ax, yb - ay) >= min_gap_m
    if not ok.any():
        ok = mask
    inner = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1] * COSTMAP_RES_M  # [m] to the region edge
    d = np.where(ok, np.hypot(xb - target_xy[0], yb - target_xy[1]) - 3.0 * np.minimum(inner, 0.5), np.inf)
    r, c = np.unravel_index(np.argmin(d), d.shape)
    return int(r), int(c)


def path_points(p: dict) -> list[tuple[float, float]]:
    """Vehicle + planned waypoints in body frame, densified every 0.25 m (for marker avoidance)."""
    pts = [(0.0, 0.0)] + [to_body(p["pose"], w[0], w[1]) for w in p.get("waypoints") or []]
    dense = []
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / 0.25))
        dense += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(n + 1)]
    return dense or pts


def side_word(x: float, y: float) -> str:
    fb = "ahead" if x > 0.5 else ("behind" if x < -0.5 else "")
    lr = "left" if y > 0.5 else ("right" if y < -0.5 else "")
    return "-".join(w for w in (fb, lr) if w) or "at the rover"


# ================================================================ per-moment analysis (markers, notes)
def analyse(k: int, seed: int, t: float, extent: int) -> dict:
    rows = telemetry(seed)
    p, n_applied = packet_at(rows, t)
    codes = decode_costmap(p["costmap_u4_hex"])
    res = json.loads((run_dir(seed) / "result.json").read_text())
    dbg = debug_tick(seed, p["t"])
    trans = supervisor_transitions(seed)
    xb, yb = cell_body_xy()
    wp_body = [to_body(p["pose"], w[0], w[1]) for w in p.get("waypoints") or []]
    wp_codes = []
    for bx, by in wp_body:
        r, c = int((8 - bx) / COSTMAP_RES_M), int((8 - by) / COSTMAP_RES_M)
        wp_codes.append(int(codes[r, c]) if 0 <= r < COSTMAP_N and 0 <= c < COSTMAP_N else -1)
    avoid = path_points(p)
    counts = {name: int((codes == v).sum()) for name, v in U4.items()}
    counts["GROUND"] = int(((codes >= 1) & (codes <= 8)).sum())
    corridor = (np.abs(yb) < 0.6) & (xb > 0.4) & (xb < 5.0)  # 1.2 m wide, bumper .. 5 m ahead
    corridor_nearest = {name: round(float(xb[corridor & (codes == v)].min()), 2)
                        for name, v in U4.items() if (corridor & (codes == v)).any()}
    markers, notes, facts = [], [], {}
    a, t_r = float(dbg["a_mps2"]), float(dbg["t_r_s"])
    v_meas = float(p["speed_mps"])

    if k == 1:
        crest = components(codes, (U4["CREST_SHADOW"],))[0]
        ditch_all = components(codes, (U4["DITCH"],))
        ditch = ditch_all[0]
        dm = codes == U4["DITCH"]
        ditch_span = {"x_min_m": round(float(xb[dm].min()), 2), "x_max_m": round(float(xb[dm].max()), 2),
                      "y_min_m": round(float(yb[dm].min()), 2), "y_max_m": round(float(yb[dm].max()), 2),
                      "nearest_m": round(float(np.hypot(xb[dm], yb[dm]).min()), 2), "n_cells": int(dm.sum()),
                      "n_components": len(ditch_all)}
        # second leader: the confirmed-ditch region nearest the rover
        near_ditch = min(ditch_all[1:] or ditch_all, key=lambda d: d["nearest_m"])
        cand = components(codes, (U4["DITCH_CANDIDATE"],))[0]
        n_wp_crest = sum(1 for c in wp_codes if c == U4["CREST_SHADOW"])
        d_stop = stop_distance(v_meas, a, t_r)
        facts = {"crest_shadow_region": {kk: crest[kk] for kk in ("n_cells", "centroid_body_m", "nearest_m")},
                 "crest_shadow_nearest_in_1.2m_corridor_m": corridor_nearest.get("CREST_SHADOW"),
                 "waypoints_in_crest_shadow": f"{n_wp_crest} of {len(wp_codes)}",
                 "confirmed_ditch_region": {kk: ditch[kk] for kk in ("n_cells", "centroid_body_m", "nearest_m")},
                 "confirmed_ditch_cells_total": counts["DITCH"], "confirmed_ditch_span_body_m": ditch_span,
                 "confirmed_ditch_region_near_rover": {kk: near_ditch[kk] for kk in ("n_cells", "centroid_body_m", "nearest_m")},
                 "ditch_candidate_region": {kk: cand[kk] for kk in ("n_cells", "centroid_body_m", "nearest_m")},
                 "stop_distance_at_measured_speed_m": round(d_stop, 2),
                 "stop_distance_inputs": {"v_mps": v_meas, "a_mps2": a, "T_r_s": t_r, "B_m": GOVERNOR_MARGIN_M},
                 "v_cap_from_r_cert_term_mps": round(dbg["gov_terms"]["r_cert"], 3)}
        # (number, cell, marker offset [px] for thin regions: the marker sits beside them with a leader)
        markers = [(1, anchor_cell(crest["mask"], tuple(crest["centroid_body_m"]), avoid, 1.0), (0, 0)),
                   (2, [anchor_cell(ditch["mask"], tuple(ditch["centroid_body_m"]), avoid, 0.6),
                        anchor_cell(near_ditch["mask"], tuple(near_ditch["centroid_body_m"]), avoid, 0.3)], (-64, 58)),
                   (3, anchor_cell(cand["mask"], tuple(cand["centroid_body_m"]), avoid, 0.6), (-44, 62))]
        notes = [
            (1, f"Crest shadow: ground beyond the crest that it has not seen yet, from "
                f"{corridor_nearest['CREST_SHADOW']:.1f} m ahead. The planned path crosses it: the planner allows "
                f"that at a cost, while speed is still governed by the seen ground."),
            (2, f"Confirmed ditch, lethal, along its {'left' if ditch_span['y_min_m'] > 0 else 'side'}: from "
                f"{-ditch_span['x_min_m']:.1f} m behind to {ditch_span['x_max_m']:.1f} m ahead, nearest "
                f"{ditch_span['nearest_m']:.1f} m (the console draws it in the same red as obstacles)."),
            (3, f"Ditch candidates, not yet confirmed, {side_word(*cand['centroid_body_m'])}."),
            (None, f"R_cert = {p['r_cert_m']:.1f} m of certified ground ahead, capped here by how far it can see "
                   f"(GOV_RVIS). Stopping from {v_meas:.2f} m/s takes about {d_stop:.1f} m (Estimated), so the "
                   f"{p['v_cap_mps']:.2f} m/s cap is not what limits speed in this frame."),
        ]
    elif k == 2:
        leth = components(codes, (U4["LETHAL"], U4["DYNAMIC"], U4["DITCH"]))
        near_r = corridor & np.isin(codes, (U4["LETHAL"], U4["DYNAMIC"], U4["DITCH"]))
        nearest_haz = float(xb[near_r].min())
        rr, cc = np.argwhere(near_r & (xb == nearest_haz))[0]
        comp = next(cp for cp in leth if cp["mask"][rr, cc])
        unseen = components(codes, (U4["UNSEEN"],))[0]
        wp_unseen = [w for w, c in zip(wp_body, wp_codes) if c == U4["UNSEEN"]]
        n_wp_unknown = sum(1 for c in wp_codes if c in (U4["UNSEEN"], U4["OCCLUDED"], U4["CREST_SHADOW"]))
        enter = next(tr for tr in trans if tr["to"] == "STOP_AND_LOOK")
        stop = res["stops"][0]
        facts = {"nearest_lethal_in_1.2m_corridor_m": round(nearest_haz, 2),
                 "lethal_region_at_marker": {kk: comp[kk] for kk in ("n_cells", "centroid_body_m", "nearest_m")},
                 "waypoints_on_unseen": f"{len(wp_unseen)} of {len(wp_codes)}",
                 "waypoints_on_unseen_or_occluded": f"{n_wp_unknown} of {len(wp_codes)}",
                 "waypoints_body_m": [[round(x, 2), round(y, 2)] for x, y in wp_body],
                 "waypoint_codes": [U4_NAME.get(c, "GROUND" if 1 <= c <= 8 else str(c)) for c in wp_codes],
                 "unseen_cells": counts["UNSEEN"], "supervisor_transition": enter, "referee_stop": stop}
        markers = [(1, anchor_cell(comp["mask"], tuple(comp["centroid_body_m"]), avoid, 0.6), (0, 0)),
                   (2, anchor_cell(unseen["mask"], wp_unseen[len(wp_unseen) // 2] if wp_unseen else tuple(unseen["centroid_body_m"]), avoid, 1.0), (0, 0))]
        notes = [
            (1, f"Obstacle cells (lethal) in its lane from {nearest_haz:.1f} m ahead."),
            (2, f"The planned route turns back and right, across ground it has not seen "
                f"({n_wp_unknown} of {len(wp_codes)} waypoints unseen or occluded)."),
            (None, f"After 3 s without progress toward B, the supervisor switched to STOP AND LOOK "
                   f"(log: {enter['t']:.2f} s; first console packet: {p['t']:.1f} s). It stops and turns in place "
                   f"+45\u00b0, \u221245\u00b0 and back to see the ground around it, then replans."),
        ]
    else:
        # where it stopped: pose of the first STOP_AND_LOOK packet, in the current body frame
        sal = [r for r in rows if r["mode"] == "STOP_AND_LOOK"]
        stop_pose = sal[0]["pose"]
        sx, sy = to_body(p["pose"], stop_pose[0], stop_pose[1])
        enter = next(tr for tr in trans if tr["to"] == "STOP_AND_LOOK")
        leave = next(tr for tr in trans if tr["from"] == "STOP_AND_LOOK")
        trail_len = float(sum(math.hypot(b["pose"][0] - a_["pose"][0], b["pose"][1] - a_["pose"][1])
                              for a_, b in zip(rows[:n_applied - 1], rows[1:n_applied])))
        facts = {"stop_point_body_m": [round(sx, 2), round(sy, 2)], "stop_point_distance_m": round(math.hypot(sx, sy), 2),
                 "look_interval_s": [enter["t"], leave["t"]], "resume_reason": leave["why"],
                 "trail_length_from_packets_m": round(trail_len, 1),
                 "waypoint_codes": [U4_NAME.get(c, "GROUND" if 1 <= c <= 8 else str(c)) for c in wp_codes],
                 "governor_binding_term": dbg["gov_binding"], "gov_terms_mps": dbg["gov_terms"]}
        markers = [(1, ("body", sx, sy), (70, -34))]
        notes = [
            (1, f"Where it stopped and looked ({enter['t']:.1f}\u2013{leave['t']:.1f} s, supervisor log). "
                f"The blue trail is the route it drove after the look."),
            (None, f"Back to NOMINAL at {leave['t']:.1f} s ('{leave['why']}'). Now {p['speed_mps']:.2f} m/s "
                   f"under a {p['v_cap_mps']:.2f} m/s cap (the platform limit), with R_cert = {p['r_cert_m']:.1f} m."),
        ]
    return {"k": k, "seed": seed, "t": t, "extent": extent, "packet": p, "n_packets_applied": n_applied, "codes": codes,
            "counts": counts, "corridor_nearest_m": corridor_nearest, "result": res, "debug": dbg, "transitions": trans,
            "markers": markers, "notes": notes, "facts": facts, "rows": rows}


# ================================================================ screenshots
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def screenshot_all(moments: list[dict]) -> tuple[list[dict], list[str]]:
    port = free_port()
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(REPO)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shots, logs = [], []
    try:
        time.sleep(1.0)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            for m in moments:
                run = f"/{RUNS}/{m['seed']:03d}"
                page = browser.new_page(viewport=VIEWPORT, device_scale_factor=DSF)
                page.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
                page.goto(f"http://127.0.0.1:{port}/operator_ui/index.html?deterministic=1"
                          f"&mission={run}/mission.json&replay={run}/autonomy/telemetry.jsonl")
                page.wait_for_function("window.__consoleReady === true", timeout=15000)
                page.wait_for_function("document.getElementById('pk-n').textContent !== '0'", timeout=15000)
                page.click(f"#zoom button[data-extent='{m['extent']}']")
                page.evaluate(f"window.consoleSeek({m['t']})")
                page.evaluate("window.dispatchEvent(new Event('resize'))")
                time.sleep(0.4)
                dom = page.evaluate("(ids) => Object.fromEntries(Object.entries(ids).map(([k, id]) => [k, document.getElementById(id).textContent]))", DOM_FIELDS)
                dom["replay_badge_visible"] = page.evaluate("!document.getElementById('replay-badge').hidden")
                dom["event_log"] = page.evaluate("Array.from(document.querySelectorAll('#log li')).map(li => li.innerText.replace(/\\n/g, ' | '))")
                boxes = page.evaluate("(sel) => Object.fromEntries(Object.entries(sel).map(([k, s]) => { const r = document.querySelector(s).getBoundingClientRect(); return [k, [r.x, r.y, r.width, r.height]]; }))", BOX_SELECTORS)
                png = page.screenshot()
                page.close()
                shots.append({"dom": dom, "boxes": boxes, "png": Image.open(io.BytesIO(png)).convert("RGB")})
            browser.close()
    finally:
        srv.terminate()
        srv.wait(timeout=5)
    return shots, logs


def crop_css(im: Image.Image, x0: float, y0: float, x1: float, y1: float, out_w: int | None = None) -> Image.Image:
    """Crop a CSS-px box from the DSF screenshot and resample to S output px per CSS px."""
    c = im.crop((round(x0 * DSF), round(y0 * DSF), round(x1 * DSF), round(y1 * DSF)))
    w = out_w if out_w is not None else round((x1 - x0) * S)
    return c.resize((w, round((y1 - y0) * S)), Image.LANCZOS)


# ================================================================ composition
def pt(px: float) -> float:
    """Font size [pt] that renders at `px` pixels at DPI."""
    return px * 72.0 / DPI


FS_NOTE, FS_KEY, FS_TICK, FS_AXIS, FS_PROV = pt(26), pt(24), pt(22), pt(23), pt(22)
LINE_SPACING = 1.15
LINE_H = 34  # px per wrapped note line at FS_NOTE x LINE_SPACING (rounded up)
NOTE_GAP = 12
NOTE_WRAP_WIDE, NOTE_WRAP_SIDE = 126, 55  # characters per line (measured: ~13.6 px per character at 26 px)
KEY_COLS, KEY_ROW_H = 4, 40


def colour_key(c: dict, p: dict) -> list[tuple]:
    """Key entries for the cell states present in this frame (console colours, contract names) + glyphs."""
    key = [("sw2", "Seen ground, low \u2192 high cost", (CELL_RGB["GROUND"], CELL_RGB["OLIVE"])),
           ("sw", "Unseen", CELL_RGB["UNSEEN"])]
    if c["OCCLUDED"]:
        key.append(("sw", "Occluded (unseen)", CELL_RGB["OCCLUDED"]))
    if c["CREST_SHADOW"]:
        key.append(("sw", "Crest shadow (unseen)", CELL_RGB["CREST_SHADOW"]))
    if c["DITCH_CANDIDATE"]:
        key.append(("sw", "Ditch candidate", CELL_RGB["DITCH_CANDIDATE"]))
    if c["LETHAL"] or c["DITCH"]:
        key.append(("sw", "Lethal or confirmed ditch" if c["DITCH"] else "Lethal obstacle", CELL_RGB["LETHAL"]))
    if c["DYNAMIC"]:
        key.append(("sw", "Newly occupied (lethal)", CELL_RGB["DYNAMIC"]))
    if c["WATER"]:
        key.append(("sw", "Water / mud", CELL_RGB["WATER"]))
    if p.get("waypoints"):
        key.append(("path", "Planned path", CONSOLE_PLAN))
    key.append(("trail", "Trail driven", CONSOLE_TRAIL))
    return key


def compose(m: dict, shot: dict) -> tuple[Path, Path, dict]:
    im, bx = shot["png"], shot["boxes"]
    p = m["packet"]
    # --- banner: left part (mode chip + reason) + right part (IN MODE, DIST TO B), empty middle removed
    b_x, b_y, b_w, b_h = bx["banner"]
    r0 = bx["banner_right0"][0] - 18  # left edge of the IN MODE readout, minus the banner gap
    right_w_css = (b_x + b_w) - r0
    left_w_css = BANNER_W / S - right_w_css
    banner_l = crop_css(im, b_x, b_y, b_x + left_w_css, b_y + b_h)
    banner_r = crop_css(im, r0, b_y, b_x + b_w, b_y + b_h)
    # --- costmap: panel header (left part) + centre crop of the canvas
    h_x, h_y, h_w, h_h = bx["map_head"]
    head = crop_css(im, h_x + 1, h_y, h_x + 1 + MAP_CROP_W_CSS, h_y + h_h, out_w=COL_L_W)
    c_x, c_y, c_w, c_h = bx["map"]
    cx_css = c_x + c_w / 2
    map_x0 = cx_css - MAP_CROP_W_CSS / 2
    mapc = crop_css(im, map_x0, c_y, map_x0 + MAP_CROP_W_CSS, c_y + c_h, out_w=COL_L_W)
    # --- governor panel: header + speed and R_cert bars (sparkline excluded)
    g_x, g_y, g_w, _ = bx["gov"]
    t_x, t_y, t_w, t_h = bx["rcert_ticks"]
    gov = crop_css(im, g_x, g_y, g_x + g_w, t_y + t_h + 12, out_w=COL_R_W)

    num_notes = [(n, t) for n, t in m["notes"] if n is not None]
    side_notes = [t for n, t in m["notes"] if n is None]
    # keep a number with its unit when wrapping (no-break space; textwrap only breaks on ASCII whitespace)
    keep = lambda t: re.sub(r"(\d) (m/s|m|s)\b", "\\1\u00a0\\2", t)  # noqa: E731
    num_lines = [textwrap.wrap(keep(t), NOTE_WRAP_WIDE) for _, t in num_notes]
    side_lines = [textwrap.wrap(keep(t), NOTE_WRAP_SIDE) for t in side_notes]
    key = colour_key(m["counts"], p)
    key_rows = math.ceil(len(key) / KEY_COLS)

    y_banner = 14
    y_row2 = y_banner + banner_l.height + 22
    map_top = y_row2 + head.height
    col_l_h = head.height + mapc.height
    y_notes = y_row2 + col_l_h + 26
    notes_h = sum(len(ls) * LINE_H for ls in num_lines) + max(0, len(num_lines) - 1) * NOTE_GAP
    y_key = y_notes + notes_h + 22
    y_prov = y_key + key_rows * KEY_ROW_H + 12
    fig_h = y_prov + 36

    fig = plt.figure(figsize=(FIG_W / DPI, fig_h / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(fig_h, 0)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    def place(img: Image.Image, x: float, y: float) -> None:
        ax.imshow(np.asarray(img), extent=(x, x + img.width, y + img.height, y), interpolation="none", zorder=1)

    place(banner_l, M_L, y_banner)
    place(banner_r, M_L + BANNER_W - banner_r.width, y_banner)
    # close the banner box across the removed middle (console border colour)
    for yy in (y_banner + 1, y_banner + banner_l.height - 1):
        ax.plot([M_L + banner_l.width - 1, M_L + BANNER_W - banner_r.width + 1], [yy, yy], color=CONSOLE_BORDER, lw=1.2, zorder=2)
    place(head, M_L, y_row2)
    place(mapc, M_L, map_top)
    ax.add_patch(Rectangle((M_L, y_row2), COL_L_W, col_l_h, fill=False, ec=CONSOLE_BORDER, lw=1.5, zorder=3))
    place(gov, X_R, y_row2)
    ax.add_patch(Rectangle((X_R, y_row2), COL_R_W, gov.height, fill=False, ec=CONSOLE_BORDER, lw=1.5, zorder=3))

    # --- costmap geometry (app.js renderMap): track-up, vehicle at the canvas centre
    ppm_css = min(c_w, c_h) / m["extent"]
    half_css = COSTMAP_N * COSTMAP_RES_M / 2 * ppm_css
    cy_css = c_h / 2

    def body_to_fig(xb_m: float, yb_m: float) -> tuple[float, float]:
        cxp, cyp = c_w / 2 - yb_m * ppm_css, cy_css - xb_m * ppm_css  # canvas CSS
        return M_L + (cxp - (c_w / 2 - MAP_CROP_W_CSS / 2)) * S, map_top + cyp * S

    def cell_to_fig(r: int, c: int) -> tuple[float, float]:
        cxp = c_w / 2 - half_css + (c + 0.5) * COSTMAP_RES_M * ppm_css
        cyp = cy_css - half_css + (r + 0.5) * COSTMAP_RES_M * ppm_css
        return M_L + (cxp - (c_w / 2 - MAP_CROP_W_CSS / 2)) * S, map_top + cyp * S

    marker_px = {}
    for num, where, (ox, oy) in m["markers"]:
        wl = where if isinstance(where, list) else [where]
        anchors = [body_to_fig(w[1], w[2]) if w[0] == "body" else cell_to_fig(*w) for w in wl]
        fx, fy = anchors[0][0] + ox, anchors[0][1] + oy
        marker_px[num] = {"anchors": [(round(a0, 1), round(a1, 1)) for a0, a1 in anchors], "marker": (round(fx, 1), round(fy, 1))}
        for i_a, (ax_, ay_) in enumerate(anchors):
            if not (ox or oy) and i_a == 0:
                continue
            # leader from the marker to the anchor cell (white casing so it reads on any cell colour)
            ax.plot([fx, ax_], [fy, ay_], color=WHITE, lw=5, solid_capstyle="round", zorder=5)
            ax.plot([fx, ax_], [fy, ay_], color=INK, lw=2, solid_capstyle="round", zorder=5)
            ax.add_patch(Circle((ax_, ay_), 6, fc=INK, ec=WHITE, lw=1.8, zorder=6))
        ax.add_patch(Circle((fx, fy), 22, fc=NAVY, ec=WHITE, lw=2.4, zorder=6))
        ax.text(fx, fy + 1, str(num), color=WHITE, fontsize=pt(26), fontweight="bold", ha="center", va="center", zorder=7)

    # scale bar (bottom-right of the costmap crop), same length rule as app.js drawScale
    L_m = 2 if m["extent"] <= 16 else 5
    bar_px = L_m * ppm_css * S
    sx0, sy0 = M_L + COL_L_W - 24 - bar_px, map_top + mapc.height - 24
    ax.add_patch(FancyBboxPatch((sx0 - 12, sy0 - 44), bar_px + 24, 58, boxstyle="round,pad=0,rounding_size=6",
                                fc=WHITE, ec=CONSOLE_BORDER, lw=1, alpha=0.95, zorder=6))
    ax.plot([sx0, sx0 + bar_px], [sy0, sy0], color=INK, lw=2.5, solid_capstyle="butt", zorder=7)
    for xx in (sx0, sx0 + bar_px):
        ax.plot([xx, xx], [sy0 - 8, sy0 + 5], color=INK, lw=2, zorder=7)
    ax.text(sx0 + bar_px / 2, sy0 - 10, f"{L_m} m", color=INK, fontsize=pt(23), ha="center", va="bottom", zorder=7)

    # --- right column: speed / v_cap over the whole run (same packets the console sparkline plots)
    side_h = sum(len(ls) * LINE_H for ls in side_lines) + max(0, len(side_lines) - 1) * NOTE_GAP
    y_side = y_row2 + col_l_h - side_h
    y_title = y_row2 + gov.height + 24
    period = float(np.median(np.diff([r["t"] for r in m["rows"]])))
    ttl = ax.text(X_R + 4, y_title, "Speed and v_cap over this run", color=INK, fontsize=FS_KEY, ha="left", va="top", fontweight="bold")
    ttl_w = ttl.get_window_extent(renderer=fig.canvas.get_renderer()).width * FIG_W / fig.bbox.width
    ax.text(X_R + 4 + ttl_w + 10, y_title + 1, f"\u00b7 one packet every {period:.1f} s", color=INK, fontsize=FS_KEY, ha="left", va="top")
    ax_top = y_title + 44
    ax_bot = y_side - 30 - 62  # room for tick labels + x label
    ax_l, ax_r = X_R + 78, X_R + COL_R_W - 14
    cax = fig.add_axes([ax_l / FIG_W, 1 - ax_bot / fig_h, (ax_r - ax_l) / FIG_W, (ax_bot - ax_top) / fig_h])
    rows = m["rows"]
    tt = np.array([r["t"] for r in rows])
    sp = np.array([r["speed_mps"] for r in rows])
    vc = np.array([r["v_cap_mps"] for r in rows])
    t_end = float(tt[-1])
    for tr0, tr1 in look_intervals(m["transitions"], m["result"]["time"]):
        cax.axvspan(tr0, tr1, color=RED, alpha=0.10, lw=0, zorder=0)
        cax.text((tr0 + tr1) / 2, 1.05, "stop\nand\nlook", color=INK, fontsize=FS_TICK, ha="center", va="center",
                 linespacing=1.05, zorder=6)
    cax.step(tt, vc, where="post", color=GREY_BASE, lw=2, zorder=2)
    cax.plot(tt, sp, color=NAVY, lw=2, zorder=3, solid_joinstyle="round", solid_capstyle="round")
    cax.axvline(p["t"], color=INK, lw=1.2, zorder=4)
    cax.plot([p["t"]], [p["speed_mps"]], "o", ms=8, mfc=NAVY, mec=WHITE, mew=2, zorder=5)
    right_side = p["t"] < 0.7 * t_end
    cax.text(p["t"] + (0.012 if right_side else -0.012) * t_end, 2.42, f"shown: {p['t']:.1f} s", color=INK, fontsize=FS_TICK,
             ha="left" if right_side else "right", va="center", zorder=6)
    cax.set_xlim(0, t_end)
    cax.set_ylim(0, 2.65)
    cax.set_yticks([0, 1, 2])
    cax.set_xlabel("Mission time (s)", fontsize=FS_AXIS, color=INK, labelpad=4)
    cax.set_ylabel("Speed (m/s)", fontsize=FS_AXIS, color=INK, labelpad=6)
    cax.tick_params(labelsize=FS_TICK, colors=INK, length=3)
    cax.grid(axis="y", color=GRID, lw=1)
    cax.set_axisbelow(True)
    for sp_name in ("top", "right"):
        cax.spines[sp_name].set_visible(False)
    for sp_name in ("left", "bottom"):
        cax.spines[sp_name].set_color("#BDBDBD")
    # direct labels (no legend box): v_cap above its line near the right end, speed under its line
    cax.text(0.985 * t_end, 2.06, "v_cap", color=INK, fontsize=FS_TICK, ha="right", va="bottom", zorder=6)
    tail = tt > 0.86 * t_end
    cax.text(0.985 * t_end, float(sp[tail].min()) - 0.1, "speed", color=INK, fontsize=FS_TICK, ha="right", va="top", zorder=6)

    # --- side note (right column bottom): mode / governor reading of this frame
    y = y_side
    for ls in side_lines:
        ax.add_patch(Rectangle((X_R + 4, y + 3), 6, len(ls) * LINE_H - 8, fc=NAVY, ec="none", zorder=6))
        ax.text(X_R + 24, y, "\n".join(ls), color=INK, fontsize=FS_NOTE, ha="left", va="top", linespacing=LINE_SPACING)
        y += len(ls) * LINE_H + NOTE_GAP

    # --- numbered notes (full width, under the costmap)
    y = y_notes
    for (num, _), ls in zip(num_notes, num_lines):
        ax.add_patch(Circle((M_L + 18, y + 16), 18, fc=NAVY, ec="none", zorder=6))
        ax.text(M_L + 18, y + 17, str(num), color=WHITE, fontsize=pt(23), fontweight="bold", ha="center", va="center", zorder=7)
        ax.text(M_L + 48, y, "\n".join(ls), color=INK, fontsize=FS_NOTE, ha="left", va="top", linespacing=LINE_SPACING)
        y += len(ls) * LINE_H + NOTE_GAP

    # --- colour key (console colours, honest names); only states present in this frame + glyphs
    col_w = BANNER_W / KEY_COLS
    for i, (kind, label, col) in enumerate(key):
        kx = M_L + (i % KEY_COLS) * col_w
        ky = y_key + (i // KEY_COLS) * KEY_ROW_H
        if kind == "sw":
            ax.add_patch(Rectangle((kx, ky + 5), 26, 26, fc=np.array(col) / 255, ec="none"))
            tx = kx + 36
        elif kind == "sw2":
            ax.add_patch(Rectangle((kx, ky + 5), 26, 26, fc=np.array(col[0]) / 255, ec="none"))
            ax.add_patch(Rectangle((kx + 28, ky + 5), 26, 26, fc=np.array(col[1]) / 255, ec="none"))
            tx = kx + 64
        elif kind == "path":
            ax.plot([kx, kx + 36], [ky + 18, ky + 18], color=col, lw=3, solid_capstyle="round")
            ax.add_patch(Circle((kx + 36, ky + 18), 6, fc=WHITE, ec=INK, lw=2, zorder=5))
            tx = kx + 52
        elif kind == "rover":
            ax.add_patch(Rectangle((kx + 6, ky + 2), 16, 30, fc=np.array(col) / 255, ec="none"))
            tx = kx + 36
        else:
            ax.plot([kx, kx + 40], [ky + 18, ky + 18], color=col, lw=3, solid_capstyle="round")
            tx = kx + 50
        ax.text(tx, ky + 18, label, color=INK, fontsize=FS_KEY, ha="left", va="center")
    res = m["result"]
    prov = (f"Simulated, tier-0, EVAL seed {m['seed']}, FULL \u00b7 console replay (not live) of "
            f"{RUNS}/{m['seed']:03d}/autonomy/telemetry.jsonl")
    ax.text(M_L, y_prov + 16, prov, color=MUTED, fontsize=FS_PROV, ha="left", va="center")

    png, svg = OUT / f"console_{m['k']}.png", OUT / f"console_{m['k']}.svg"
    fig.savefig(png, dpi=DPI, facecolor=WHITE, bbox_inches="tight", pad_inches=0)
    fig.savefig(svg, dpi=DPI, facecolor=WHITE, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    layout = {"banner_crop_css": {"left": [b_x, b_y, round(left_w_css, 1), b_h], "right": [round(r0, 1), b_y, round(right_w_css, 1), b_h]},
              "costmap_crop_css": [round(map_x0, 1), c_y, MAP_CROP_W_CSS, c_h], "governor_crop_css": [g_x, g_y, g_w, round(t_y + t_h + 12 - g_y, 1)],
              "scale_output_px_per_css_px": S, "marker_px": marker_px, "provenance_line": prov}
    return png, svg, layout


def look_intervals(trans: list[dict], t_end: float) -> list[tuple[float, float]]:
    """STOP_AND_LOOK intervals from the supervisor log."""
    out, t0 = [], None
    for tr in trans:
        if tr["to"] == "STOP_AND_LOOK":
            t0 = tr["t"]
        elif tr["from"] == "STOP_AND_LOOK" and t0 is not None:
            out.append((t0, tr["t"]))
            t0 = None
    if t0 is not None:
        out.append((t0, t_end))
    return out


# ================================================================ context stats (for captions / honesty)
def eval_context() -> dict:
    """STOP_AND_LOOK occurrence and governor-binding frequency across all 60 FULL EVAL runs."""
    sal, n_nom, n_bind = [], 0, 0
    for d in sorted((REPO / RUNS).glob("0*")):
        res = json.loads((d / "result.json").read_text())
        modes = {json.loads(l)["mode"] for l in (d / "autonomy" / "telemetry.jsonl").read_text().splitlines() if l.strip()}
        if "STOP_AND_LOOK" in modes:
            sal.append({"seed": res["seed"], "family": res["family"], "success": res["success"], "failure_type": res["failure_type"],
                        "stops": res["stops"]})
        for f in sorted((d / "autonomy" / "debug").glob("tick_*.npz")):
            ex = json.loads(str(np.load(f)["extras_json"]))
            if ex.get("mode") == "NOMINAL":
                n_nom += 1
                n_bind += ex.get("gov_binding") in ("r_cert", "r_vis")
    return {"stop_and_look_runs": sal, "n_runs_with_stop_and_look": len(sal),
            "n_of_those_reached_B": sum(1 for s in sal if s["success"]),
            "outcomes_of_the_others": sorted({f"{s['seed']}: {s['failure_type']}" for s in sal if not s["success"]}),
            "governor_binding_nominal_ticks": {"seen_range_terms_binding": n_bind, "nominal_ticks": n_nom,
                                               "fraction": round(n_bind / max(n_nom, 1), 3),
                                               "how": "5 Hz per-tick debug records of all 60 FULL EVAL runs, mode NOMINAL, "
                                                      "gov_binding in (r_cert, r_vis)"},
            "source": f"{RUNS}/<seed>/result.json, autonomy/telemetry.jsonl, autonomy/debug/tick_*.npz"}


# ================================================================ main
def main() -> int:
    moments = [analyse(*mm) for mm in MOMENTS]
    shots, logs = screenshot_all(moments)
    ctx = eval_context()
    records = []
    for m, shot in zip(moments, shots):
        png, svg, layout = compose(m, shot)
        p, res, dbg = m["packet"], m["result"], m["debug"]
        with Image.open(png) as im:
            size = list(im.size)
        records.append({
            "file": str(png.relative_to(REPO)), "svg": str(svg.relative_to(REPO)), "size_px": size,
            "label": "Simulated", "seed": m["seed"], "family": res["family"], "split": res["split"],
            "sensor_mode": res["sensor_mode"], "seek_t_s": m["t"], "view_extent_m": m["extent"],
            "costmap_extent_m": COSTMAP_N * COSTMAP_RES_M,
            "sources": {"mission": f"{RUNS}/{m['seed']:03d}/mission.json", "telemetry": f"{RUNS}/{m['seed']:03d}/autonomy/telemetry.jsonl",
                        "debug_tick": f"{RUNS}/{m['seed']:03d}/autonomy/debug/tick_{dbg['tick']:06d}.npz",
                        "autonomy_log": f"{RUNS}/{m['seed']:03d}/autonomy/autonomy.log", "result": f"{RUNS}/{m['seed']:03d}/result.json"},
            "telemetry_packet_shown": {"t": p["t"], "seq": p["seq"], "mode": p["mode"], "reason": p["reason"], "pose": p["pose"],
                                       "pos_sigma_m": p["pos_sigma_m"], "v_cap_mps": p["v_cap_mps"], "r_cert_m": p["r_cert_m"],
                                       "speed_mps": p["speed_mps"], "packet_bytes": p.get("packet_bytes"),
                                       "n_packets_applied": m["n_packets_applied"]},
            "debug_tick_shown": {k: dbg[k] for k in ("tick", "t", "reason", "gov_terms", "gov_binding", "a_mps2", "t_r_s", "r_path_m", "r_vis_m", "cmd_v", "speed_meas")},
            "console_readout_visible": {k: shot["dom"][k] for k in ("mode", "reason", "in_mode", "dist_to_b", "speed_vcap", "r_cert")},
            "console_readout_not_shown": {k: v for k, v in shot["dom"].items() if k not in ("mode", "reason", "in_mode", "dist_to_b", "speed_vcap", "r_cert")},
            "cell_counts_in_frame": m["counts"], "corridor_nearest_m": m["corridor_nearest_m"],
            "annotation_facts": m["facts"], "notes_drawn": [txt for _, txt in m["notes"]],
            "supervisor_transitions": m["transitions"],
            "run_outcome": {kk: res[kk] for kk in ("success", "failure_type", "time", "path_length", "spl", "final_error", "stops", "false_stops")},
            "chart": {"what": "speed_mps and v_cap_mps of every telemetry packet of the run (what the console sparkline plots); "
                              "red band = STOP_AND_LOOK interval from autonomy.log",
                      "n_packets": len(m["rows"]), "packet_period_s": round(float(np.median(np.diff([r['t'] for r in m['rows']]))), 3)},
            "layout": layout,
        })
    (OUT / "console.json").write_text(json.dumps({
        "generator": "deck_assets/final/src/console_shots.py",
        "what": "Crops of the operator console (operator_ui/, unmodified) replaying logged tier-0 EVAL closed-loop runs, "
                "plus a deck-style annotation layer (numbered markers, notes, chart, colour key). Replay only, not a live link.",
        "viewport_css_px": VIEWPORT, "device_scale_factor": DSF,
        "caveats": [
            "Replay of a logged simulation run, not live; the console is replay-only today (docs/PIPELINE_STATUS.md).",
            "Tier-0 synthetic depth sensor: no camera images, so VO cannot run (the console's VO tile reads LOST; "
            "localisation is wheel + gyro, docs/SIMULATION.md).",
            "Omitted panels: the integrity gauge shows a fixed q = 1.00 / p_fail 0.00 in tier-0 (metagross/autonomy/node.py sets "
            "q, p_fail = 1.0, 0.0 without images; the learned monitor is not running); link kbps / loss are computed by app.js "
            "from logged packet sizes (no radio in replay); the GO/HOLD/RESUME/E-STOP buttons are not wired to the autonomy.",
            "Telemetry cadence in these logs is one packet per 0.6 s (1.67 Hz: every 3rd 5 Hz frame; node.py sends when "
            ">= 0.5 s has passed), not the 2 Hz the console's downlink subtitle says; console times are packet times.",
            "Console colours are operator_ui's own tokens (royal-blue plan/trail #1D4ED8/#2563EB, red lethal, orange 'Dynamic'); "
            "the annotation layer uses the deck palette. The console draws confirmed ditch (code 14) in the same red as "
            "obstacles (code 15); its legend word 'Dynamic' means 'newly occupied where free space was observed' "
            "(contracts/messages.py), not a tracked moving object.",
            "Frames 2 and 3 are one selected successful example: STOP_AND_LOOK occurred in "
            f"{ctx['n_runs_with_stop_and_look']} of 60 FULL EVAL runs and {ctx['n_of_those_reached_B']} of those reached B.",
            "Per-run numbers shown here are not yet in results/claims.csv (that ledger is generated; see claims_to_register).",
        ],
        "eval_context": ctx,
        "figures": records,
        "claims_to_register": claims_rows(records, ctx),
        "browser_errors": logs,
    }, indent=1, default=float))
    for r in records:
        print(r["file"], r["size_px"], r["telemetry_packet_shown"]["mode"], r["telemetry_packet_shown"]["reason"])
    print("\n".join(logs) or "no page errors")
    return 0


def claims_rows(records: list[dict], ctx: dict) -> list[dict]:
    """Rows in results/claims.csv format (id,value,unit,label,source,note) for the numbers in these figures."""
    out = []
    for r in records:
        s, p, o = r["seed"], r["telemetry_packet_shown"], r["run_outcome"]
        src = r["sources"]["telemetry"]
        note = (f"EVAL closed loop, sensor mode tier0, held-out seed {s} ({r['family']}); one selected example shown in the "
                f"operator console replay; packet seq {p['seq']}")
        base = f"console_s{s:03d}_t{str(p['t']).replace('.', 'p')}"
        out += [
            {"id": f"{base}_speed_mps", "value": p["speed_mps"], "unit": "m/s", "label": "Simulated", "source": f"{src}#seq={p['seq']}.speed_mps", "note": note},
            {"id": f"{base}_v_cap_mps", "value": p["v_cap_mps"], "unit": "m/s", "label": "Simulated", "source": f"{src}#seq={p['seq']}.v_cap_mps", "note": note},
            {"id": f"{base}_r_cert_m", "value": p["r_cert_m"], "unit": "m", "label": "Simulated", "source": f"{src}#seq={p['seq']}.r_cert_m", "note": note},
        ]
        f = r["annotation_facts"]
        if "stop_distance_at_measured_speed_m" in f:
            out.append({"id": f"{base}_stop_distance_m", "value": f["stop_distance_at_measured_speed_m"], "unit": "m", "label": "Estimated",
                        "source": f"{r['sources']['debug_tick']} (a_mps2, t_r_s) + defaults.GOVERNOR_MARGIN_M",
                        "note": "d_stop = v^2/(2a) + v*T_r + B at the logged measured speed"})
        if "nearest_lethal_in_1.2m_corridor_m" in f:
            out.append({"id": f"{base}_nearest_lethal_ahead_m", "value": f["nearest_lethal_in_1.2m_corridor_m"], "unit": "m", "label": "Simulated",
                        "source": f"{src}#seq={p['seq']}.costmap_u4_hex", "note": "decoded 4-bit costmap, 1.2 m wide lane ahead of the bumper"})
        if "crest_shadow_nearest_in_1.2m_corridor_m" in f:
            out.append({"id": f"{base}_crest_shadow_ahead_m", "value": f["crest_shadow_nearest_in_1.2m_corridor_m"], "unit": "m", "label": "Simulated",
                        "source": f"{src}#seq={p['seq']}.costmap_u4_hex", "note": "decoded 4-bit costmap, 1.2 m wide lane ahead of the bumper"})
        for tr in r["supervisor_transitions"]:
            out.append({"id": f"console_s{s:03d}_mode_{tr['from'].lower()}_to_{tr['to'].lower()}_t_s", "value": tr["t"], "unit": "s", "label": "Simulated",
                        "source": f"{r['sources']['autonomy_log']}", "note": tr["why"]})
        out += [{"id": f"console_s{s:03d}_outcome_time_s", "value": o["time"], "unit": "s", "label": "Simulated", "source": f"{r['sources']['result']}#time", "note": f"success={o['success']}"},
                {"id": f"console_s{s:03d}_outcome_spl", "value": o["spl"], "unit": "", "label": "Simulated", "source": f"{r['sources']['result']}#spl", "note": ""}]
    out += [{"id": "eval_full_runs_with_stop_and_look", "value": ctx["n_runs_with_stop_and_look"], "unit": "runs of 60", "label": "Simulated",
             "source": ctx["source"], "note": "FULL, tier-0, EVAL seeds 0-59"},
            {"id": "eval_full_stop_and_look_runs_reached_B", "value": ctx["n_of_those_reached_B"], "unit": "runs", "label": "Simulated",
             "source": ctx["source"], "note": "of the runs that entered STOP_AND_LOOK"}]
    uniq = {}
    for row in out:
        uniq.setdefault(row["id"], row)
    return list(uniq.values())


if __name__ == "__main__":
    sys.exit(main())
